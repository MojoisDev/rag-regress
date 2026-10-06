"""Deterministic word-based corpus chunking."""

from rag_regress.config import ChunkingConfig
from rag_regress.errors import UserInputError
from rag_regress.hashing import hash_canonical
from rag_regress.models import Chunk, CorpusSnapshot, Document


def chunk_corpus(corpus: CorpusSnapshot, config: ChunkingConfig) -> tuple[Chunk, ...]:
    """Split corpus documents into overlapping word chunks."""
    chunks: list[Chunk] = []
    step = config.size - config.overlap
    for document in corpus.documents:
        words = document.content.split()
        for sequence, start in enumerate(range(0, len(words), step)):
            end = min(start + config.size, len(words))
            text = " ".join(words[start:end])
            if text:
                chunks.append(_make_chunk(document, sequence, start, end, text, config))
    if not chunks:
        raise UserInputError("Corpus produced no non-empty chunks")
    return tuple(chunks)


def _make_chunk(
    document: Document,
    sequence: int,
    start_word: int,
    end_word: int,
    text: str,
    config: ChunkingConfig,
) -> Chunk:
    chunk_id = hash_canonical(
        {
            "document_id": document.id,
            "sequence": sequence,
            "start_word": start_word,
            "end_word": end_word,
            "chunking": config.model_dump(mode="json"),
        }
    )
    return Chunk(
        id=chunk_id,
        document_id=document.id,
        document_path=document.relative_path,
        sequence=sequence,
        text=text,
        start_word=start_word,
        end_word=end_word,
    )
