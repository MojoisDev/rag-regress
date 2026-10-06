from pathlib import Path

import pytest

from rag_regress.config import CorpusConfig
from rag_regress.corpus import load_corpus, normalize_text
from rag_regress.errors import UserInputError


def test_corpus_is_sorted_normalized_and_stable(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("Beta  \r\ntext\r\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Alpha\n\n", encoding="utf-8")
    config = CorpusConfig(path=tmp_path, include=["**/*.txt", "**/*.md"])

    first = load_corpus(config)
    second = load_corpus(config)

    assert [doc.relative_path for doc in first.documents] == ["a.txt", "b.md"]
    assert first.documents[1].content == "Beta\ntext"
    assert first.fingerprint == second.fingerprint
    assert normalize_text("a  \r\n\r\n") == "a"


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UserInputError, match="No supported documents"):
        load_corpus(CorpusConfig(path=tmp_path, include=["**/*.md"]))


def test_absolute_include_pattern_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UserInputError, match="absolute"):
        load_corpus(CorpusConfig(path=tmp_path, include=["/tmp/*.md"]))


def test_symlink_escaping_corpus_is_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("private", encoding="utf-8")
    (corpus / "escape.md").symlink_to(outside)

    with pytest.raises(UserInputError, match="outside corpus root"):
        load_corpus(CorpusConfig(path=corpus, include=["**/*.md"]))
