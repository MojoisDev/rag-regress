import sys
import types

import numpy as np
import pytest

from rag_regress.config import EmbeddingConfig
from rag_regress.embeddings import build_embedder, normalize_vectors
from rag_regress.errors import ExecutionError


def test_normalize_vectors_returns_float32_unit_rows() -> None:
    result = normalize_vectors(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64))

    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(2))


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[float("nan"), 1.0]], dtype=np.float32),
        np.array([[float("inf"), 1.0]], dtype=np.float32),
        np.array([], dtype=np.float32),
    ],
)
def test_invalid_vectors_are_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(ExecutionError, match="embedding"):
        normalize_vectors(vectors)


def test_deterministic_test_embedder_returns_normalized_vectors() -> None:
    from tests.helpers import DeterministicTestEmbedder

    embedder = DeterministicTestEmbedder()

    vectors = embedder.embed_documents(["alpha beta", "beta gamma"])

    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
    assert embedder.embed_query("alpha").shape == (1, 4)


def test_sentence_transformer_adapter_normalizes_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_embedding_dimension(self) -> int:
            return 2

        def get_sentence_embedding_dimension(self) -> int:
            raise AssertionError("deprecated dimension API must not be called")

        def encode(
            self, texts: list[str], *, convert_to_numpy: bool, show_progress_bar: bool
        ) -> np.ndarray:
            assert convert_to_numpy is True
            assert show_progress_bar is False
            return np.array([[3.0, 4.0] for _ in texts], dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = build_embedder(
        EmbeddingConfig(provider="sentence_transformers", model="test-model", normalize=True)
    )
    vectors = embedder.embed_documents(["first", "second"])

    assert embedder.dimension == 2
    assert embedder.metadata.model == "test-model"
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2))


def test_sentence_transformer_adapter_supports_legacy_dimension_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacySentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=LegacySentenceTransformer),
    )

    embedder = build_embedder(
        EmbeddingConfig(provider="sentence_transformers", model="legacy-model", normalize=True)
    )

    assert embedder.dimension == 2


def _skip_if_model_not_cached(error: ExecutionError) -> None:
    if "couldn't find them in the cached files" not in str(error):
        raise error
    pytest.skip("model is not cached locally", allow_module_level=False)


def test_model_cache_skip_does_not_hide_other_execution_errors() -> None:
    error = ExecutionError("Unable to load embedding model 'test': invalid embedding dimension")

    with pytest.raises(ExecutionError, match="invalid embedding dimension"):
        _skip_if_model_not_cached(error)


@pytest.mark.model
def test_sentence_transformer_embedder_smoke_uses_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = EmbeddingConfig(
        provider="sentence_transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
    )

    try:
        embedder = build_embedder(config)
    except ExecutionError as exc:
        _skip_if_model_not_cached(exc)

    vector = embedder.embed_query("How can a user reset their password?")

    assert np.all(np.isfinite(vector))
    np.testing.assert_allclose(np.linalg.norm(vector, axis=1), np.ones(1))
