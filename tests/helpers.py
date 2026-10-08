"""Test-only helpers for deterministic, offline embedding tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from rag_regress.embeddings import EmbeddingMetadata, normalize_vectors


class DeterministicTestEmbedder:
    """Hash tokens into four normalized dimensions without loading a model."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(provider="test", model="deterministic-four-dim", revision="a" * 40)

    @property
    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.split():
                vectors[row, self._dimension(token)] += 1.0
        return normalize_vectors(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])

    @staticmethod
    def _dimension(token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big") % 4
