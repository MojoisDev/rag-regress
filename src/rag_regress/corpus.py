"""Safe, deterministic loading of supported corpus documents."""

import hashlib
from pathlib import Path

from rag_regress.config import CorpusConfig
from rag_regress.errors import UserInputError
from rag_regress.hashing import hash_canonical
from rag_regress.models import CorpusSnapshot, Document

_SUPPORTED_SUFFIXES = {".md", ".txt"}


def normalize_text(text: str) -> str:
    """Normalize line endings and non-semantic outer and trailing whitespace."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip(" \t") for line in lines).strip()


def load_corpus(config: CorpusConfig) -> CorpusSnapshot:
    """Load all configured corpus files in deterministic, safe path order."""
    root = config.path.resolve()
    if not root.is_dir():
        raise UserInputError(f"Corpus root is not a directory: {root}")

    paths = _discover_paths(root, config.include)
    if not paths:
        raise UserInputError(f"No supported documents found beneath corpus root: {root}")

    documents = tuple(_load_document(root, path) for path in paths)
    manifest = [
        {
            "id": document.id,
            "relative_path": document.relative_path,
            "checksum": document.checksum,
        }
        for document in documents
    ]
    return CorpusSnapshot(documents=documents, fingerprint=hash_canonical(manifest))


def _discover_paths(root: Path, include: list[str]) -> list[Path]:
    discovered: dict[str, Path] = {}
    for pattern in include:
        if Path(pattern).is_absolute():
            raise UserInputError(f"Corpus include pattern must not be absolute: {pattern}")
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise UserInputError(f"Document resolves outside corpus root: {candidate}") from exc
            if candidate.suffix.lower() not in _SUPPORTED_SUFFIXES:
                raise UserInputError(f"Unsupported corpus document suffix: {candidate.suffix}")
            relative_key = candidate.relative_to(root).as_posix()
            discovered[relative_key] = candidate
    return [discovered[key] for key in sorted(discovered)]


def _load_document(root: Path, path: Path) -> Document:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UserInputError(f"Document resolves outside corpus root: {path}") from exc
    relative_path = path.relative_to(root).as_posix()

    try:
        content = normalize_text(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise UserInputError(f"Unable to read corpus document {path}: {exc}") from exc

    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    document_id = hash_canonical({"relative_path": relative_path, "checksum": checksum})
    return Document(
        id=document_id,
        relative_path=relative_path,
        title=Path(relative_path).stem,
        content=content,
        checksum=checksum,
    )
