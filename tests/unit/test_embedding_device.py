import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from rag_regress.config import EmbeddingConfig, index_config_fingerprint, load_pipeline_config
from rag_regress.embeddings import SentenceTransformerEmbedder
from rag_regress.errors import ExecutionError
from rag_regress.hashing import hash_canonical


def _config(device: str | None = "auto") -> EmbeddingConfig:
    return EmbeddingConfig.model_validate(
        {
            "provider": "sentence_transformers",
            "model": "test-model",
            "revision": "a" * 40,
            "normalize": True,
            "device": device,
        }
    )


def test_default_device_preserves_existing_index_identity() -> None:
    from pathlib import Path

    config = load_pipeline_config(Path(__file__).parents[2] / "examples/configs/baseline.yaml")
    assert config.embedding.device == "auto"
    legacy_embedding = config.embedding.model_dump(mode="json", exclude={"device"})
    legacy_identity = hash_canonical(
        {
            "schema_version": 1,
            "chunking": config.chunking.model_dump(mode="json"),
            "embedding": legacy_embedding,
            "metric": "cosine",
        }
    )
    assert index_config_fingerprint(config) == legacy_identity
    fingerprints = {legacy_identity}
    for device in ("cpu", "cuda"):
        changed = config.model_copy(
            update={
                "embedding": _config(device).model_copy(
                    update={"model": config.embedding.model, "revision": config.embedding.revision}
                )
            }
        )
        fingerprints.add(index_config_fingerprint(changed))
    assert len(fingerprints) == 3


@pytest.mark.parametrize("device", ["gpu", "cuda:1", "", None])
def test_invalid_device_is_rejected(device: str | None) -> None:
    with pytest.raises(ValidationError, match="device"):
        _config(device)


@pytest.mark.parametrize(
    "requested,actual",
    [("auto", "cpu"), ("cpu", "cpu"), ("cuda", "cuda:0"), ("auto", "cuda:0"), ("auto", "mps:0")],
)
def test_device_selection_and_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    actual: str,
) -> None:
    class Model:
        def __init__(self, name: str, *, revision: str, device: str | None = None) -> None:
            assert device == (None if requested == "auto" else requested)
            self.device = torch.device(actual)

        def get_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=Model,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: actual.startswith("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    embedder = SentenceTransformerEmbedder(_config(requested))
    assert embedder.embed_query("question").shape == (1, 2)
    runtime = embedder.runtime
    assert runtime.device == actual
    assert runtime.gpu_name == ("Test GPU" if actual.startswith("cuda") else None)
    assert runtime.torch_version == torch.__version__
    assert runtime.cuda_version == torch.version.cuda


def test_requested_cuda_unavailable_fails_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected model download")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=must_not_load,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ExecutionError, match="CUDA.*unavailable"):
        SentenceTransformerEmbedder(_config("cuda"))


def test_requested_cuda_cannot_silently_fall_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: SimpleNamespace(
                device=torch.device("cpu"),
                get_embedding_dimension=lambda: 2,
            ),
        ),
    )
    with pytest.raises(ExecutionError, match="requested.*cuda.*cpu"):
        SentenceTransformerEmbedder(_config("cuda"))
