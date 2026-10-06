"""Immutable domain models for corpus artifacts."""

from pydantic import BaseModel, ConfigDict


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
