"""Validated FAISS IndexFlatIP bundles with atomic, content-addressed persistence."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from numpy.typing import NDArray

from rag_regress.artifacts import atomic_write_json
from rag_regress.config import PipelineConfig, index_config_fingerprint
from rag_regress.embeddings import Embedder, normalize_vectors
from rag_regress.errors import ExecutionError, UserInputError
from rag_regress.models import Chunk, CorpusSnapshot, IndexBundleManifest

_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.json"
_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class LoadedIndexBundle:
    """A validated index bundle ready for deterministic retrieval."""

    manifest: IndexBundleManifest
    index: FaissIndex
    chunks: tuple[Chunk, ...]


class FaissIndex:
    """A small checked wrapper around FAISS' exact inner-product index."""

    def __init__(self, raw: Any, dimension: int) -> None:
        self._raw = raw
        self.dimension = dimension

    @classmethod
    def build(cls, vectors: NDArray[np.generic]) -> FaissIndex:
        """Build an exact, normalized inner-product index from document vectors."""
        validated = validate_index_vectors(vectors)
        raw = faiss.IndexFlatIP(validated.shape[1])
        raw.add(validated)
        return cls(raw, int(validated.shape[1]))

    def search(
        self, query: NDArray[np.generic], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return descending scores and deterministic index positions for one query."""
        if top_k <= 0:
            raise ExecutionError("top_k must be positive")
        validated = validate_query_vector(query, self.dimension)
        limit = min(top_k, int(self._raw.ntotal))
        if limit <= 0:
            raise ExecutionError("Cannot search an empty FAISS index")
        scores, positions = self._raw.search(validated.reshape(1, -1), limit)
        flat_scores = cast(NDArray[np.float32], scores[0])
        flat_positions = cast(NDArray[np.int64], positions[0])
        if np.any(flat_positions < 0):
            raise ExecutionError("FAISS search returned an invalid index position")
        order = np.lexsort((flat_positions, -flat_scores))
        return flat_scores[order], flat_positions[order]

    def save(self, path: Path) -> None:
        """Save this index to its bundle-local filename."""
        try:
            faiss.write_index(self._raw, str(path))
        except Exception as exc:
            raise ExecutionError(f"Unable to write FAISS index {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path, expected_dimension: int) -> FaissIndex:
        """Load and validate an exact inner-product index with the expected dimension."""
        try:
            raw = faiss.read_index(str(path))
        except Exception as exc:
            raise ExecutionError(f"Unable to load FAISS index {path}: {exc}") from exc
        if not isinstance(raw, faiss.IndexFlatIP):
            raise ExecutionError(f"FAISS index must be IndexFlatIP: {path}")
        dimension = int(raw.d)
        if expected_dimension <= 0 or dimension != expected_dimension:
            raise ExecutionError(
                "FAISS index dimension "
                f"{dimension} does not match expected {expected_dimension}: {path}"
            )
        if int(raw.ntotal) <= 0:
            raise ExecutionError(f"FAISS index is empty: {path}")
        return cls(raw, dimension)


def validate_index_vectors(vectors: NDArray[np.generic]) -> NDArray[np.float32]:
    """Validate and normalize a non-empty, two-dimensional document matrix."""
    return normalize_vectors(vectors)


def validate_query_vector(
    query: NDArray[np.generic], expected_dimension: int
) -> NDArray[np.float32]:
    """Validate and normalize one query vector against an index dimension."""
    try:
        array = np.asarray(query, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExecutionError("Invalid query vector: unable to convert to float32") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] != expected_dimension:
        raise ExecutionError(
            "Invalid query vector: expected one vector with the index embedding dimension"
        )
    return cast(NDArray[np.float32], normalize_vectors(array)[0])


def bundle_path(config: PipelineConfig, corpus_fingerprint: str) -> Path:
    """Return the deterministic storage location for one corpus/configuration pair."""
    return (
        config.storage.directory
        / "indexes"
        / corpus_fingerprint
        / index_config_fingerprint(config)
    )


def build_index_bundle(
    config: PipelineConfig,
    corpus: CorpusSnapshot,
    chunks: tuple[Chunk, ...],
    embedder: Embedder,
) -> IndexBundleManifest:
    """Build, validate, and atomically publish a content-addressed FAISS bundle."""
    destination = bundle_path(config, corpus.fingerprint)
    if destination.exists():
        return load_index_bundle(config, corpus.fingerprint).manifest
    if not chunks:
        raise ExecutionError("Cannot build an index bundle with no chunks")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.mkdir()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        index = FaissIndex.build(vectors)
        if index.dimension != embedder.dimension:
            raise ExecutionError(
                "Embedding output dimension does not match embedder metadata: "
                f"{index.dimension} != {embedder.dimension}"
            )
        index.save(temporary / _INDEX_FILENAME)
        atomic_write_json(
            temporary / _CHUNKS_FILENAME,
            {"chunks": [chunk.model_dump(mode="json") for chunk in chunks]},
        )
        manifest = IndexBundleManifest(
            corpus_fingerprint=corpus.fingerprint,
            index_config_fingerprint=index_config_fingerprint(config),
            embedding=embedder.metadata,
            dimension=index.dimension,
            chunk_count=len(chunks),
            chunk_ids=tuple(chunk.id for chunk in chunks),
            index_file=_INDEX_FILENAME,
            chunks_file=_CHUNKS_FILENAME,
        )
        atomic_write_json(temporary / _MANIFEST_FILENAME, manifest.model_dump(mode="json"))
        _load_index_bundle_at(temporary, config, corpus.fingerprint)
        temporary.replace(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_index_bundle(config: PipelineConfig, corpus_fingerprint: str) -> LoadedIndexBundle:
    """Load the exact bundle requested by corpus and index configuration identities."""
    destination = bundle_path(config, corpus_fingerprint)
    if not destination.exists():
        raise UserInputError(
            f"Index bundle not found: {destination}. Run `rag-regress ingest` before evaluation."
        )
    if not destination.is_dir():
        raise ExecutionError(f"Invalid index bundle target (expected directory): {destination}")
    return _load_index_bundle_at(destination, config, corpus_fingerprint)


def _load_index_bundle_at(
    directory: Path, config: PipelineConfig, corpus_fingerprint: str
) -> LoadedIndexBundle:
    try:
        manifest = IndexBundleManifest.model_validate(_read_json(directory / _MANIFEST_FILENAME))
        _validate_manifest_identity(manifest, config, corpus_fingerprint, directory)
        _validate_relative_filename(manifest.index_file, _INDEX_FILENAME, directory)
        _validate_relative_filename(manifest.chunks_file, _CHUNKS_FILENAME, directory)
        chunks = _load_chunks(directory / manifest.chunks_file)
        _validate_chunk_table(chunks, manifest, directory)
        index = FaissIndex.load(directory / manifest.index_file, manifest.dimension)
        if int(index._raw.ntotal) != len(chunks):
            raise ExecutionError(
                f"FAISS index vector count does not match chunk count in bundle: {directory}"
            )
    except (OSError, ValueError, TypeError) as exc:
        raise ExecutionError(f"Invalid index bundle {directory}: {exc}") from exc
    return LoadedIndexBundle(manifest=manifest, index=index, chunks=chunks)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunks(path: Path) -> tuple[Chunk, ...]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != {"chunks"}:
        raise ExecutionError(f"Chunk table must contain only a chunks field: {path}")
    raw_chunks = payload["chunks"]
    if not isinstance(raw_chunks, list):
        raise ExecutionError(f"Chunk table chunks field must be an array: {path}")
    return tuple(Chunk.model_validate(item) for item in raw_chunks)


def _validate_manifest_identity(
    manifest: IndexBundleManifest,
    config: PipelineConfig,
    corpus_fingerprint: str,
    directory: Path,
) -> None:
    if manifest.corpus_fingerprint != corpus_fingerprint:
        raise ExecutionError(f"Bundle corpus fingerprint does not match its path: {directory}")
    if manifest.index_config_fingerprint != index_config_fingerprint(config):
        raise ExecutionError(
            f"Bundle index configuration fingerprint does not match its path: {directory}"
        )
    if manifest.dimension <= 0:
        raise ExecutionError(f"Bundle embedding dimension must be positive: {directory}")
    if manifest.chunk_count <= 0 or manifest.chunk_count != len(manifest.chunk_ids):
        raise ExecutionError(f"Bundle chunk metadata is inconsistent: {directory}")


def _validate_relative_filename(filename: str, expected: str, directory: Path) -> None:
    candidate = Path(filename)
    if filename != expected or candidate.name != filename or candidate.is_absolute():
        raise ExecutionError(f"Bundle filename is invalid: {directory}")


def _validate_chunk_table(
    chunks: tuple[Chunk, ...], manifest: IndexBundleManifest, directory: Path
) -> None:
    if len(chunks) != manifest.chunk_count:
        raise ExecutionError(f"Bundle chunk count does not match manifest: {directory}")
    if tuple(chunk.id for chunk in chunks) != manifest.chunk_ids:
        raise ExecutionError(f"Bundle chunk ordering does not match manifest: {directory}")
