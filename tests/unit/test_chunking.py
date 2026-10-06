import pytest

from rag_regress.chunking import chunk_corpus
from rag_regress.config import ChunkingConfig
from rag_regress.errors import UserInputError
from rag_regress.models import CorpusSnapshot, Document


def _corpus(content: str) -> CorpusSnapshot:
    document = Document(
        id="doc-1",
        relative_path="guide.md",
        title="guide",
        content=content,
        checksum="sum-1",
    )
    return CorpusSnapshot(documents=(document,), fingerprint="corpus-1")


def test_word_chunks_overlap_and_preserve_boundaries() -> None:
    chunks = chunk_corpus(
        _corpus("one two three four five six seven"),
        ChunkingConfig(strategy="words", size=4, overlap=1),
    )

    assert [(chunk.text, chunk.start_word, chunk.end_word) for chunk in chunks] == [
        ("one two three four", 0, 4),
        ("four five six seven", 3, 7),
        ("seven", 6, 7),
    ]


def test_chunk_ids_are_stable_and_change_with_configuration() -> None:
    corpus = _corpus("one two three four five")
    first = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=3, overlap=1))
    second = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=3, overlap=1))
    changed = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=4, overlap=1))

    assert [item.id for item in first] == [item.id for item in second]
    assert [item.id for item in first] != [item.id for item in changed]


def test_empty_corpus_content_is_rejected() -> None:
    with pytest.raises(UserInputError, match="Corpus produced no non-empty chunks"):
        chunk_corpus(_corpus(" \n\t "), ChunkingConfig(strategy="words", size=3, overlap=1))
