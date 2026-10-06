"""Immutable domain models for corpus and persisted index artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_regress.embeddings import EmbeddingMetadata


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


class IndexBundleManifest(BaseModel):
    """Versioned metadata for one content-addressed FAISS bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    corpus_fingerprint: str
    index_config_fingerprint: str
    embedding: EmbeddingMetadata
    dimension: int
    chunk_count: int
    chunk_ids: tuple[str, ...]
    index_file: str
    chunks_file: str
