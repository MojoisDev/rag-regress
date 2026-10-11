import re

import pytest
from pydantic import ValidationError

from rag_regress.chunking import chunk_corpus
from rag_regress.config import ChunkingConfig
from rag_regress.embeddings import EmbeddingMetadata, Tokenization
from rag_regress.errors import UserInputError
from rag_regress.models import Chunk
from tests.unit.test_chunking import _corpus


class OffsetTokenizer:
    is_fast = True

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        offsets = [match.span() for match in re.finditer(r"\w+|[^\w\s]", text)]
        ids = list(range(len(offsets) + (2 if add_special_tokens else 0)))
        return {"input_ids": ids, "offset_mapping": offsets}


def tokenization(tokenizer=None, max_length=8, revision="a" * 40):
    return Tokenization(
        tokenizer or OffsetTokenizer(),
        max_length,
        EmbeddingMetadata("test", "tokenizer", revision),
    )


def test_token_chunks_preserve_source_and_overlap():
    text = "café, 你好!\n\tthree four five"
    config = ChunkingConfig(strategy="tokens", size=4, overlap=1)
    chunks = chunk_corpus(_corpus(text), config, tokenization())
    assert [chunk.text for chunk in chunks] == ["café, 你好!", "!\n\tthree four five"]
    assert [(chunk.start_token, chunk.end_token) for chunk in chunks] == [(0, 4), (3, 7)]
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.start_word is chunk.end_word is None
        assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert chunks == chunk_corpus(_corpus(text), config, tokenization())
    assert (
        chunks[0].id != chunk_corpus(_corpus(text), config, tokenization(revision="b" * 40))[0].id
    )


def test_token_mode_requires_capability_and_reserves_special_tokens():
    config = ChunkingConfig(strategy="tokens", size=7, overlap=0)
    with pytest.raises(UserInputError, match="tokenizer"):
        chunk_corpus(_corpus("hello"), config)
    with pytest.raises(UserInputError, match="6"):
        chunk_corpus(_corpus("hello"), config, tokenization())


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, -1), (4, 4)])
def test_direct_chunk_config_rejects_invalid_windows(size, overlap):
    with pytest.raises(ValidationError):
        ChunkingConfig(strategy="tokens", size=size, overlap=overlap)


@pytest.mark.parametrize("offsets", [[(0, 0)], [(0, 99)], [(2, 3), (0, 1)]])
def test_unusable_offsets_fail_clearly(offsets):
    class BrokenTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    with pytest.raises(UserInputError, match="offset"):
        chunk_corpus(
            _corpus("one two"),
            ChunkingConfig(strategy="tokens", size=4, overlap=0),
            tokenization(BrokenTokenizer()),
        )


def test_standalone_subword_expansion_shrinks_window():
    class SubwordTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text == "three four five":
                result["input_ids"].append(99)
            return result

    chunks = chunk_corpus(
        _corpus("one two three four five"),
        ChunkingConfig(strategy="tokens", size=3, overlap=1),
        tokenization(SubwordTokenizer()),
    )
    assert [chunk.text for chunk in chunks] == ["one two three", "three four", "four five"]


def test_empty_token_corpus_rejected():
    with pytest.raises(UserInputError, match="no non-empty chunks"):
        chunk_corpus(
            _corpus(" \n"), ChunkingConfig(strategy="tokens", size=4, overlap=0), tokenization()
        )


def test_word_ids_and_legacy_chunk_shape_are_unchanged():
    chunks = chunk_corpus(
        _corpus("one two three four five"), ChunkingConfig(strategy="words", size=3, overlap=1)
    )
    assert [chunk.id for chunk in chunks] == [
        "3b2b0a82046e0c2f317e2228c07bd2f0bd211b9c96959c09400d9f416656d7b6",
        "79b951161499926ca4e24bee13939ffab4b169f5b9e6615347d8ef11d54cb061",
        "662f8aa6bebb1c2a591feae8d00fe61f22f6b8511cfd7fc86f5283d0d5d46420",
    ]
    for chunk in chunks:
        payload = chunk.model_dump(mode="json", exclude_none=True)
        assert set(payload) == {
            "id",
            "document_id",
            "document_path",
            "sequence",
            "text",
            "start_word",
            "end_word",
        }
        assert Chunk.model_validate(payload) == chunk


@pytest.mark.parametrize(
    "change",
    [
        {"start_token": None},
        {"end_token": 0},
        {"end_char": 9},
        {"start_word": 0},
    ],
)
def test_token_boundaries_are_validated(change):
    chunk = chunk_corpus(
        _corpus("one two"), ChunkingConfig(strategy="tokens", size=4, overlap=0), tokenization()
    )[0]
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk.model_dump() | change)


def test_overlap_that_cannot_advance_fails_and_slow_tokenizer_is_rejected():
    class ExpandingTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text == "two three":
                result["input_ids"].append(99)
            return result

    with pytest.raises(UserInputError, match="preserving overlap"):
        chunk_corpus(
            _corpus("one two three four"),
            ChunkingConfig(strategy="tokens", size=2, overlap=1),
            tokenization(ExpandingTokenizer()),
        )
    slow = OffsetTokenizer()
    slow.is_fast = False
    with pytest.raises(UserInputError, match="fast tokenizer"):
        tokenization(slow).validate_size(4)


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_missing_model_input_limit_rejected(limit):
    with pytest.raises(UserInputError, match="input limit"):
        tokenization(max_length=limit).validate_size(4)


def test_unavailable_offsets_and_tokenizer_errors_are_clear():
    class NoOffsets(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

    class FailingTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            raise ValueError("unavailable offsets")

    for tokenizer in [NoOffsets(), FailingTokenizer()]:
        with pytest.raises(UserInputError, match="offsets"):
            chunk_corpus(
                _corpus("one"),
                ChunkingConfig(strategy="tokens", size=4, overlap=0),
                tokenization(tokenizer),
            )


def test_model_preprocessing_that_changes_text_is_rejected():
    from types import SimpleNamespace

    from rag_regress.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder._metadata = tokenization().metadata
    embedder._model = SimpleNamespace(
        tokenizer=OffsetTokenizer(),
        max_seq_length=8,
        _first_module=lambda: SimpleNamespace(do_lower_case=True),
    )
    with pytest.raises(UserInputError, match="preprocessing"):
        _ = embedder.tokenization


def test_model_default_prompt_and_missing_tokenizer_are_rejected():
    from types import SimpleNamespace

    from rag_regress.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder._model = SimpleNamespace(default_prompt_name="document")
    with pytest.raises(UserInputError, match="default prompt"):
        _ = embedder.tokenization
    embedder._model = SimpleNamespace()
    with pytest.raises(UserInputError, match="no tokenizer"):
        _ = embedder.tokenization


def test_budget_also_checks_stripped_model_input():
    class WhitespaceTokenizer(OffsetTokenizer):
        def __call__(self, text, *, add_special_tokens=False, **kwargs):
            count = 1 if text.startswith(" ") else 2
            return {"input_ids": [1] * (count + (2 if add_special_tokens else 0))}

    assert not tokenization(WhitespaceTokenizer()).fits(" word", 1)
