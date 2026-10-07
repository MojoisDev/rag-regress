"""Immutable domain models for corpus, evaluation, and persisted artifacts."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_regress.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from rag_regress.embeddings import EmbeddingMetadata
from rag_regress.hashing import hash_canonical


class Document(BaseModel):
    """A normalized corpus document with a stable identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    relative_path: str
    title: str
    content: str
    checksum: str


class CorpusSnapshot(BaseModel):
    """The ordered corpus and its deterministic fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[Document, ...]
    fingerprint: str


class Chunk(BaseModel):
    """A deterministic word-range from a corpus document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    document_id: str
    document_path: str
    sequence: int
    text: str
    start_word: int
    end_word: int


class RetrievalResult(BaseModel):
    """One ranked chunk returned by retrieval for an evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    document_path: str
    rank: int = Field(ge=1)
    score: float
    text: str


class EvaluationCase(BaseModel):
    """One labelled question with document and optional evidence expectations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_documents: tuple[str, ...] = Field(min_length=1)
    expected_text: tuple[str, ...] = ()

    @field_validator("id", "question")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Normalize required human-readable fields and reject blank values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("expected_documents")
    @classmethod
    def normalize_expected_documents(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize corpus-relative paths and reject duplicate targets."""
        normalized = tuple(_normalize_document_path(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("expected_documents must not contain duplicate paths")
        return normalized

    @field_validator("expected_text")
    @classmethod
    def strip_expected_text(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize optional evidence targets while preserving their order."""
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("expected_text must not contain blank values")
        return normalized


class EvaluationDataset(BaseModel):
    """The strict, versioned collection of labelled evaluation questions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        """Normalize the dataset display name and reject blank values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @property
    def fingerprint(self) -> str:
        """Return the stable identity of the canonical evaluation dataset content."""
        return hash_canonical(self.model_dump(mode="json"))


class IndexBundleManifest(BaseModel):
    """Versioned metadata for one content-addressed FAISS bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    corpus_fingerprint: str
    index_config_fingerprint: str
    embedding: EmbeddingMetadata
    dimension: int
    chunk_count: int
    chunk_ids: tuple[str, ...]
    index_file: str
    chunks_file: str


class EnvironmentInfo(BaseModel):
    """Versions of the runtime that produced one evaluation artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str
    platform: str
    numpy_version: str
    faiss_version: str
    sentence_transformers_version: str
    rag_regress_version: str


class PipelineSnapshot(BaseModel):
    """Retrieval-relevant configuration without machine-specific paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_include_patterns: tuple[str, ...]
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig


class CaseMetrics(BaseModel):
    """Document and optional evidence metrics for one labelled question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    evidence_hit: float | None = Field(default=None, ge=0.0, le=1.0)


class CaseResult(BaseModel):
    """Complete evaluated retrieval result for one dataset case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    expected_documents: tuple[str, ...]
    expected_text: tuple[str, ...]
    results: tuple[RetrievalResult, ...]
    metrics: CaseMetrics
    retrieval_ms: float = Field(ge=0.0)


class AggregateMetrics(BaseModel):
    """Aggregate quality and monotonic retrieval-latency measurements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    evidence_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    p50_retrieval_ms: float = Field(ge=0.0)
    p95_retrieval_ms: float = Field(ge=0.0)


class MetricComparison(BaseModel):
    """Baseline, candidate, and candidate-minus-baseline values for one metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: float | None
    candidate: float | None
    delta: float | None


class AggregateMetricComparison(BaseModel):
    """Metric comparisons for one complete evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: MetricComparison
    hit_rate: MetricComparison
    mrr: MetricComparison
    evidence_hit_rate: MetricComparison
    p50_retrieval_ms: MetricComparison
    p95_retrieval_ms: MetricComparison


class CaseComparison(BaseModel):
    """Per-case metric changes between two compatible evaluation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    recall_at_k: MetricComparison
    hit_at_k: MetricComparison
    reciprocal_rank: MetricComparison
    evidence_hit: MetricComparison
    retrieval_ms: MetricComparison


class ComparisonReport(BaseModel):
    """Stable comparison of compatible baseline and candidate runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: AggregateMetricComparison
    case_deltas: tuple[CaseComparison, ...]
    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_case_ids: tuple[str, ...]


class GateFailure(BaseModel):
    """One quality gate whose inclusive boundary was not met."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: str
    expected: str
    actual: float | None
    baseline: float | None = None


class GateReport(BaseModel):
    """Complete quality-gate outcome, retaining every failed gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    failures: tuple[GateFailure, ...]


class RunArtifact(BaseModel):
    """Versioned, self-describing output from one retrieval evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    tool_version: str
    metric_definition_version: Literal["1"] = "1"
    dataset_name: str
    corpus_fingerprint: str
    dataset_fingerprint: str
    pipeline: PipelineSnapshot
    environment: EnvironmentInfo
    embedding: EmbeddingMetadata
    started_at: datetime
    duration_ms: float = Field(ge=0.0)
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    metrics: AggregateMetrics
    cases: tuple[CaseResult, ...]


def _normalize_document_path(value: str) -> str:
    """Return a safe, POSIX-normalized relative corpus path."""
    normalized = value.strip()
    path = PurePosixPath(normalized.replace("\\", "/"))
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("expected document path must be relative and must not contain '..'")
    result = path.as_posix()
    if result == ".":
        raise ValueError("expected document path must not be empty")
    return result
