"""Strict loading and identity helpers for pipeline configuration files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from rag_regress.errors import UserInputError
from rag_regress.hashing import hash_canonical
from rag_regress.validation import StrictModel


class CorpusConfig(StrictModel):
    """Document corpus configuration."""

    path: Path
    include: list[str]


class ChunkingConfig(StrictModel):
    """Document chunking configuration."""

    strategy: Literal["words"]
    size: int
    overlap: int


class EmbeddingConfig(StrictModel):
    """Embedding provider configuration."""

    provider: Literal["sentence_transformers"]
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    normalize: bool
    device: Literal["auto", "cpu", "cuda"] = "auto"


class RetrievalConfig(StrictModel):
    """Nearest-neighbour retrieval configuration."""

    metric: Literal["cosine"]
    top_k: int
    relevance_threshold: float | None


class StorageConfig(StrictModel):
    """Local artifact storage configuration."""

    directory: Path


class LatencyConfig(StrictModel):
    """Unmeasured warm-up count and measured retrievals per evaluation case."""

    warmup_queries: int = Field(default=1, ge=0, strict=True)
    repetitions: int = Field(default=1, ge=1, strict=True)


class PipelineConfig(StrictModel):
    """The complete version 1 pipeline configuration."""

    schema_version: int
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    latency: LatencyConfig = Field(default_factory=LatencyConfig)

    @model_validator(mode="after")
    def validate_supported_configuration(self) -> "PipelineConfig":
        """Enforce the v0.1 constraints that span configuration fields."""
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.chunking.size <= 0:
            raise ValueError("chunking.size must be positive")
        if not 0 <= self.chunking.overlap < self.chunking.size:
            raise ValueError("chunking.overlap must satisfy 0 <= overlap < size")
        if not self.embedding.normalize:
            raise ValueError("embedding.normalize must be true for cosine retrieval")
        if self.retrieval.top_k <= 0:
            raise ValueError("retrieval.top_k must be positive")
        return self


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and validate a configuration, resolving paths relative to its file."""
    source = path.resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        config = PipelineConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise UserInputError(f"Invalid pipeline config {source}: {exc}") from exc

    base = source.parent
    return config.model_copy(
        update={
            "corpus": config.corpus.model_copy(
                update={"path": (base / config.corpus.path).resolve()}
            ),
            "storage": config.storage.model_copy(
                update={"directory": (base / config.storage.directory).resolve()}
            ),
        }
    )


def index_config_fingerprint(config: PipelineConfig) -> str:
    """Return the retrieval-relevant, location-independent configuration identity."""
    embedding = config.embedding.model_dump(mode="json")
    if config.embedding.device == "auto":
        # Preserve existing bundle paths for configs written before device selection.
        embedding.pop("device")
    identity = {
        "schema_version": config.schema_version,
        "chunking": config.chunking.model_dump(mode="json"),
        "embedding": embedding,
        "metric": config.retrieval.metric,
    }
    return hash_canonical(identity)
