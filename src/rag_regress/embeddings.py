"""Validated embedding boundary for retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from rag_regress.config import EmbeddingConfig
from rag_regress.errors import ExecutionError


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """Identity of the embedding implementation recorded with a run."""

    provider: str
    model: str


class Embedder(Protocol):
    """Provider-neutral source of normalized document and query vectors."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Return the provider and model identity."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension."""
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed document texts into normalized row vectors."""
        raise NotImplementedError

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query as a normalized single-row vector."""
        raise NotImplementedError


def normalize_vectors(vectors: NDArray[np.generic]) -> NDArray[np.float32]:
    """Validate and L2-normalize a non-empty matrix of embedding vectors."""
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExecutionError("Invalid embedding vectors: unable to convert to float32") from exc

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ExecutionError(
            "Invalid embedding vectors: expected a non-empty two-dimensional array"
        )
    if not np.all(np.isfinite(matrix)):
        raise ExecutionError("Invalid embedding vectors: values must be finite")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ExecutionError("Invalid embedding vectors: rows must have positive finite norms")

    normalized = (matrix / norms).astype(np.float32, copy=False)
    return cast(NDArray[np.float32], normalized)


class SentenceTransformerEmbedder:
    """Sentence Transformers adapter that validates every model output."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._metadata = EmbeddingMetadata(provider=config.provider, model=config.model)
        try:
            import sentence_transformers

            self._model: Any = sentence_transformers.SentenceTransformer(config.model)
            get_dimension = getattr(self._model, "get_embedding_dimension", None)
            if callable(get_dimension):
                dimension = get_dimension()
            else:
                dimension = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise ExecutionError(f"Unable to load embedding model {config.model!r}: {exc}") from exc

        if not isinstance(dimension, int) or dimension <= 0:
            raise ExecutionError(
                f"Unable to load embedding model {config.model!r}: invalid embedding dimension"
            )
        self._dimension = dimension

    @property
    def metadata(self) -> EmbeddingMetadata:
        return self._metadata

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(texts)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._encode([text])

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        expected_rows = len(texts)
        if expected_rows == 0:
            raise ExecutionError(
                f"Invalid embedding output from model {self._config.model!r}: no texts"
            )
        try:
            output = self._model.encode(
                list(texts), convert_to_numpy=True, show_progress_bar=False
            )
            vectors = np.asarray(output, dtype=np.float32)
        except Exception as exc:
            raise ExecutionError(
                f"Unable to encode with embedding model {self._config.model!r}: {exc}"
            ) from exc

        if vectors.ndim == 1 and expected_rows == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ExecutionError(
                f"Invalid embedding output from model {self._config.model!r}: unexpected row count"
            )
        if vectors.shape[1] != self._dimension:
            raise ExecutionError(
                f"Invalid embedding output from model {self._config.model!r}: unexpected dimension"
            )
        return normalize_vectors(vectors)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Build the configured v0.1 embedding implementation."""
    return SentenceTransformerEmbedder(config)
