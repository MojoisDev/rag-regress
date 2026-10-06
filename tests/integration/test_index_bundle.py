from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_regress.config import (
    ChunkingConfig,
    CorpusConfig,
    EmbeddingConfig,
    PipelineConfig,
    RetrievalConfig,
    StorageConfig,
)
from rag_regress.errors import ExecutionError
from rag_regress.index import build_index_bundle, bundle_path, load_index_bundle
from rag_regress.models import Chunk, CorpusSnapshot, Document
from tests.helpers import DeterministicTestEmbedder


def test_index_bundle_round_trips_search_and_reuses_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-a")
    embedder = DeterministicTestEmbedder()

    first = build_index_bundle(config, corpus, chunks, embedder)
    first_bundle_path = bundle_path(config, corpus.fingerprint)
    loaded = load_index_bundle(config, corpus.fingerprint)
    query = embedder.embed_query("alpha guide")
    expected_scores, expected_positions = loaded.index.search(query, top_k=2)

    original_replace = Path.replace

    def fail_replace(source: Path, destination: Path) -> Path:
        raise AssertionError("a valid content-addressed bundle must not be rewritten")

    monkeypatch.setattr(Path, "replace", fail_replace)
    second = build_index_bundle(config, corpus, chunks, embedder)
    monkeypatch.setattr(Path, "replace", original_replace)

    reloaded = load_index_bundle(config, corpus.fingerprint)
    actual_scores, actual_positions = reloaded.index.search(query, top_k=2)

    assert first == second == loaded.manifest == reloaded.manifest
    assert [loaded.chunks[position].id for position in expected_positions] == [
        reloaded.chunks[position].id for position in actual_positions
    ]
    assert actual_positions.tolist() == expected_positions.tolist()
    np.testing.assert_allclose(actual_scores, expected_scores)
    assert first_bundle_path.is_dir()


def test_failed_bundle_publication_leaves_no_final_or_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-b")
    target = bundle_path(config, corpus.fingerprint)
    original_replace = Path.replace

    def fail_final_directory_replace(source: Path, destination: Path) -> Path:
        if source.is_dir() and destination == target:
            raise OSError("simulated bundle publication interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_final_directory_replace)

    with pytest.raises(OSError, match="simulated bundle publication interruption"):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())

    assert not target.exists()
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_existing_invalid_bundle_is_rejected_without_overwriting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-c")
    target = bundle_path(config, corpus.fingerprint)
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())

    assert (target / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_existing_non_directory_bundle_target_is_invalid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus, chunks = _corpus_and_chunks("corpus-file-target")
    target = bundle_path(config, corpus.fingerprint)
    target.parent.mkdir(parents=True)
    target.write_text("not a bundle", encoding="utf-8")

    with pytest.raises(ExecutionError, match=str(target)):
        build_index_bundle(config, corpus, chunks, DeterministicTestEmbedder())


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        schema_version=1,
        corpus=CorpusConfig(path=tmp_path / "corpus", include=["**/*.md"]),
        chunking=ChunkingConfig(strategy="words", size=8, overlap=0),
        embedding=EmbeddingConfig(provider="sentence_transformers", model="test", normalize=True),
        retrieval=RetrievalConfig(metric="cosine", top_k=2, relevance_threshold=None),
        storage=StorageConfig(directory=tmp_path / "state"),
    )


def _corpus_and_chunks(fingerprint: str) -> tuple[CorpusSnapshot, tuple[Chunk, ...]]:
    first = Document(
        id="doc-alpha",
        relative_path="alpha.md",
        title="alpha",
        content="alpha guide reference",
        checksum="checksum-alpha",
    )
    second = Document(
        id="doc-beta",
        relative_path="beta.md",
        title="beta",
        content="beta handbook reference",
        checksum="checksum-beta",
    )
    from rag_regress.chunking import chunk_corpus

    corpus = CorpusSnapshot(documents=(first, second), fingerprint=fingerprint)
    return corpus, chunk_corpus(corpus, ChunkingConfig(strategy="words", size=8, overlap=0))
