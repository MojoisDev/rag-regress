from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import faiss
import numpy as np
import pytest

from rag_regress.config import (
    ChunkingConfig,
    CorpusConfig,
    EmbeddingConfig,
    PipelineConfig,
    RetrievalConfig,
    StorageConfig,
)
from rag_regress.embeddings import EmbeddingMetadata
from rag_regress.errors import ExecutionError, UserInputError
from rag_regress.index import (
    _validate_stored_vectors,
    build_index_bundle,
    bundle_path,
    load_index_bundle,
)
from rag_regress.models import Chunk, CorpusSnapshot, Document
from tests.helpers import DeterministicTestEmbedder


def test_index_bundle_round_trips_search_and_reuses_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-a")
    embedder = DeterministicTestEmbedder()

    first = build_index_bundle(config, corpus, chunks, embedder)
    first_bundle_path = bundle_path(config, corpus.fingerprint)
    loaded = load_index_bundle(config, corpus.fingerprint)
    query = embedder.embed_query("alpha guide")
    expected_scores, expected_positions = loaded.index.search(query, top_k=2)

    original_replace = Path.replace

    def fail_replace(source: Path, destination: Path) -> Path:
        raise AssertionError("a valid content-addressed bundle must not be rewritten")

    monkeypatch.setattr(Path, "replace", fail_replace)
    second = build_index_bundle(config, corpus, chunks, embedder)
    monkeypatch.setattr(Path, "replace", original_replace)

    reloaded = load_index_bundle(config, corpus.fingerprint)
    actual_scores, actual_positions = reloaded.index.search(query, top_k=2)

    assert first == second == loaded.manifest == reloaded.manifest
    assert [loaded.chunks[position].id for position in expected_positions] == [
        reloaded.chunks[position].id for position in actual_positions
    ]
    assert actual_positions.tolist() == expected_positions.tolist()
    np.testing.assert_allclose(actual_scores, expected_scores)
    assert first_bundle_path.is_dir()


def test_failed_bundle_publication_leaves_no_final_or_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-b")
    target = bundle_path(config, corpus.fingerprint)
    original_replace = Path.replace

    def fail_final_directory_replace(source: Path, destination: Path) -> Path:
        if source.is_dir() and destination == target:
            raise OSError("simulated bundle publication interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_final_directory_replace)

    with pytest.raises(OSError, match="simulated bundle publication interruption"):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())

    assert not target.exists()
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_existing_invalid_bundle_is_rejected_without_overwriting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-c")
    target = bundle_path(config, corpus.fingerprint)
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())

    assert (target / "manifest.json").read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("winner", ["valid", "invalid", "different_revision"])
def test_concurrent_publication_validates_winner_and_cleans_temporary_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner: str
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("concurrent")
    target = bundle_path(config, corpus.fingerprint)
    original_replace = Path.replace

    def publish_competitor(source: Path, destination: Path) -> Path:
        if source.is_dir() and destination == target:
            shutil.copytree(source, target)
            if winner == "invalid":
                (target / "manifest.json").write_text("{}", encoding="utf-8")
            elif winner == "different_revision":
                manifest_path = target / "manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["embedding"]["revision"] = "b" * 40
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", publish_competitor)
    if winner == "valid":
        manifest = build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
        assert manifest == load_index_bundle(config, corpus.fingerprint).manifest
    elif winner == "invalid":
        with pytest.raises(ExecutionError, match=str(target)):
            build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
        assert (target / "manifest.json").read_text(encoding="utf-8") == "{}"
    else:
        with pytest.raises(UserInputError, match="b{40}"):
            build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
        assert load_index_bundle(config, corpus.fingerprint).manifest.embedding.revision == "b" * 40
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("bad_row", [None, "nonfinite", "nonunit", "shape"])
def test_stored_vector_validation_is_bounded_and_checks_last_batch(
    tmp_path: Path, bad_row: str | None
) -> None:
    class StoredVectors:
        ntotal = 8193

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def reconstruct_n(self, start: int, count: int) -> np.ndarray:
            assert count <= 8192
            self.calls.append((start, count))
            vectors = np.zeros((count, 2), dtype=np.float32)
            vectors[:, 0] = 1
            if start == 8192:
                if bad_row == "nonfinite":
                    vectors[0, 0] = np.nan
                elif bad_row == "nonunit":
                    vectors[0, 0] = 2
                elif bad_row == "shape":
                    return vectors[:, :1]
            return vectors

    raw = StoredVectors()
    if bad_row is None:
        _validate_stored_vectors(raw, 2, tmp_path / "index.faiss")
    else:
        with pytest.raises(ExecutionError, match="shape|unit-normalized"):
            _validate_stored_vectors(raw, 2, tmp_path / "index.faiss")
    assert raw.calls == [(0, 8192), (8192, 1)]


def test_existing_non_directory_bundle_target_is_invalid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-file-target")
    target = bundle_path(config, corpus.fingerprint)
    target.parent.mkdir(parents=True)
    target.write_text("not a bundle", encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())


def test_bundle_manifest_requires_an_explicit_schema_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-missing-schema")
    build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    target = bundle_path(config, corpus.fingerprint)
    manifest_path = target / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["schema_version"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(target)):
        load_index_bundle(config, corpus.fingerprint)


def test_dangling_bundle_symlink_is_rejected_without_overwriting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-dangling-link")
    target = bundle_path(config, corpus.fingerprint)
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "missing-bundle", target_is_directory=True)

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())

    assert target.is_symlink()


def test_bundle_rejects_persisted_non_unit_faiss_vectors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-corrupt-vectors")
    build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    target = bundle_path(config, corpus.fingerprint)
    raw = faiss.read_index(str(target / "index.faiss"))
    vectors = raw.reconstruct_n(0, raw.ntotal)
    vectors[0] *= 2.0
    corrupt = faiss.IndexFlatIP(raw.d)
    corrupt.add(vectors)
    faiss.write_index(corrupt, str(target / "index.faiss"))

    with pytest.raises(ExecutionError, match="unit-normalized"):
        load_index_bundle(config, corpus.fingerprint)


def test_bundle_rejects_semantically_altered_chunk_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-corrupt-chunks")
    build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    target = bundle_path(config, corpus.fingerprint)
    chunks_path = target / "chunks.json"
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    payload["chunks"][0]["text"] = "semantically altered text with the original ID"
    chunks_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(chunks_path)):
        load_index_bundle(config, corpus.fingerprint)


def test_bundle_rejects_semantically_altered_normalized_vectors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-corrupt-normalized-vectors")
    build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    target = bundle_path(config, corpus.fingerprint)
    index_path = target / "index.faiss"
    raw = faiss.read_index(str(index_path))
    vectors = raw.reconstruct_n(0, raw.ntotal)
    vectors[0] *= -1.0
    corrupt = faiss.IndexFlatIP(raw.d)
    corrupt.add(vectors)
    faiss.write_index(corrupt, str(index_path))

    with pytest.raises(ExecutionError, match=str(index_path)):
        load_index_bundle(config, corpus.fingerprint)


def test_bundle_reuse_rejects_chunks_that_differ_from_fresh_derivation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-expected-chunk-mismatch")
    build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    changed_chunks = (
        chunks[0].model_copy(update={"text": "freshly derived text changed"}),
        *chunks[1:],
    )
    target = bundle_path(config, corpus.fingerprint)

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, changed_chunks, DeterministicTestEmbedder())


def test_bundle_reuse_rejects_different_embedder_revision(tmp_path: Path) -> None:
    class ChangedEmbedder(DeterministicTestEmbedder):
        @property
        def metadata(self) -> EmbeddingMetadata:
            return replace(super().metadata, revision="b" * 40)

    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("revision-mismatch")
    original = build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())
    with pytest.raises(UserInputError, match="b{40}"):
        build_index_bundle(config, corpus, chunks, ChangedEmbedder())
    assert load_index_bundle(config, corpus.fingerprint).manifest == original


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        schema_version=1,
        corpus=CorpusConfig(path=tmp_path / "corpus", include=["**/*.md"]),
        chunking=ChunkingConfig(strategy="words", size=8, overlap=0),
        embedding=EmbeddingConfig(
            provider="sentence_transformers", model="test", revision="a" * 40, normalize=True
        ),
        retrieval=RetrievalConfig(metric="cosine", top_k=2, relevance_threshold=None),
        storage=StorageConfig(directory=tmp_path / "state"),
    )


def _corpus_and_chunks(fingerprint: str) -> tuple[CorpusSnapshot, tuple[Chunk, ...]]:
    first = Document(
        id="doc-alpha",
        relative_path="alpha.md",
        title="alpha",
        content="alpha guide reference",
        checksum="checksum-alpha",
    )
    second = Document(
        id="doc-beta",
        relative_path="beta.md",
        title="beta",
        content="beta handbook reference",
        checksum="checksum-beta",
    )
    from rag_regress.chunking import chunk_corpus

    corpus = CorpusSnapshot(documents=(first, second), fingerprint=fingerprint)
    return corpus, chunk_corpus(corpus, ChunkingConfig(strategy="words", size=8, overlap=0))
