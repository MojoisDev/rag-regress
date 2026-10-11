"""Deterministic word and token corpus chunking."""

from dataclasses import asdict

from rag_regress.config import ChunkingConfig
from rag_regress.embeddings import Tokenization
from rag_regress.errors import UserInputError
from rag_regress.hashing import hash_canonical
from rag_regress.models import Chunk, CorpusSnapshot, Document


def chunk_corpus(
    corpus: CorpusSnapshot,
    config: ChunkingConfig,
    tokenization: Tokenization | None = None,
) -> tuple[Chunk, ...]:
    """Split documents using word windows or the embedding model's token offsets."""
    if config.strategy == "tokens":
        if tokenization is None:
            raise UserInputError("Token chunking requires the embedding model's tokenizer")
        tokenization.validate_size(config.size)
    chunks: list[Chunk] = []
    step = config.size - config.overlap
    for document in corpus.documents:
        if config.strategy == "tokens":
            assert tokenization is not None
            chunks.extend(_token_chunks(document, config, tokenization))
            continue
        words = document.content.split()
        for sequence, start in enumerate(range(0, len(words), step)):
            end = min(start + config.size, len(words))
            text = " ".join(words[start:end])
            if text:
                chunks.append(_make_chunk(document, sequence, start, end, text, config))
    if not chunks:
        raise UserInputError("Corpus produced no non-empty chunks")
    return tuple(chunks)


def _token_chunks(
    document: Document,
    config: ChunkingConfig,
    tokenization: Tokenization,
) -> list[Chunk]:
    encoded = tokenization.encode(document.content, offsets=True)
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != len(encoded["input_ids"]):
        raise UserInputError("Tokenizer returned unusable character offsets")
    previous = (0, 0)
    for span in offsets:
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= len(document.content)
            or span[0] < previous[0]
            or span[1] < previous[1]
        ):
            raise UserInputError("Tokenizer returned unusable character offsets")
        previous = (span[0], span[1])
    chunks: list[Chunk] = []
    start = 0
    while start < len(offsets):
        end = min(start + config.size, len(offsets))
        # Subword boundaries can expand when a source slice is tokenized independently.
        while end > start:
            start_char, end_char = offsets[start][0], offsets[end - 1][1]
            text = document.content[start_char:end_char]
            if tokenization.fits(text, config.size):
                break
            end -= 1
        if end == start or (end < len(offsets) and end - start <= config.overlap):
            raise UserInputError(
                "Token window cannot fit the budget while preserving overlap; "
                "increase chunking.size or reduce chunking.overlap"
            )
        boundaries = dict(
            start_token=start, end_token=end, start_char=start_char, end_char=end_char
        )
        chunk_id = hash_canonical(
            {
                "document_id": document.id,
                "sequence": len(chunks),
                **boundaries,
                "chunking": config.model_dump(mode="json"),
                "embedding": asdict(tokenization.metadata),
            }
        )
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=document.id,
                document_path=document.relative_path,
                sequence=len(chunks),
                text=text,
                **boundaries,
            )
        )
        if end == len(offsets):
            break
        start = end - config.overlap
    return chunks


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
