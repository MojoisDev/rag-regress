"""Immutable domain models for corpus, evaluation, and persisted artifacts."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    rank: int
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
