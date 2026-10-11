from pathlib import Path

import pytest

from rag_regress import experiments
from rag_regress.config import load_pipeline_config
from rag_regress.embeddings import Tokenization
from rag_regress.errors import UserInputError
from rag_regress.index import load_index_bundle
from tests.helpers import DeterministicTestEmbedder
from tests.integration.test_experiments import _write_fixture_inputs
from tests.unit.test_token_chunking import OffsetTokenizer


class TokenEmbedder(DeterministicTestEmbedder):
    @property
    def tokenization(self):
        return Tokenization(OffsetTokenizer(), 8, self.metadata)


def token_inputs(tmp_path):
    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(
        config.read_text()
        .replace("strategy: words", "strategy: tokens")
        .replace("size: 20", "size: 4")
    )
    return config, dataset


def test_tokens_ingest_reuse_evaluate_sweep(tmp_path: Path, monkeypatch):
    config, dataset = token_inputs(tmp_path)
    embedder = TokenEmbedder()
    first = experiments.ingest(config, embedder)
    assert experiments.ingest(config, embedder) == first
    bundle = load_index_bundle(load_pipeline_config(config), first.corpus_fingerprint)
    assert all(
        chunk.start_token is not None and chunk.start_word is None for chunk in bundle.chunks
    )
    run = experiments.evaluate(config, dataset, tmp_path / "run.json", embedder)
    assert run.pipeline.chunking.strategy == "tokens"
    assert len(run.cases) == 2
    monkeypatch.setattr(experiments, "build_embedder", lambda config: embedder)
    runs = experiments.sweep(config, dataset, (4, 6), tmp_path / "sweep")
    assert [run.pipeline.chunking.size for run in runs] == [4, 6]
    assert all(run.pipeline.chunking.strategy == "tokens" for run in runs)


def test_tokens_missing_capability_and_oversized_sweep_leave_no_output(tmp_path, monkeypatch):
    config, dataset = token_inputs(tmp_path)
    with pytest.raises(UserInputError, match="tokenizer"):
        experiments.ingest(config, DeterministicTestEmbedder())
    monkeypatch.setattr(experiments, "build_embedder", lambda config: TokenEmbedder())
    with pytest.raises(UserInputError, match="content-token limit"):
        experiments.sweep(config, dataset, (4, 7), tmp_path / "sweep")
    assert not (tmp_path / "sweep").exists()
    assert list(tmp_path.glob(".sweep.*.tmp")) == []


def test_evaluate_rechecks_loaded_model_limit(tmp_path):
    config, dataset = token_inputs(tmp_path)
    embedder = TokenEmbedder()
    experiments.ingest(config, embedder)

    class SmallerLimit(TokenEmbedder):
        @property
        def tokenization(self):
            return Tokenization(OffsetTokenizer(), 4, self.metadata)

    with pytest.raises(UserInputError, match="content-token limit"):
        experiments.evaluate(config, dataset, tmp_path / "run.json", SmallerLimit())
    assert not (tmp_path / "run.json").exists()
