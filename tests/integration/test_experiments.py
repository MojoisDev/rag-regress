from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_regress.errors import UserInputError
from rag_regress.experiments import evaluate, ingest
from rag_regress.models import RunArtifact
from tests.helpers import DeterministicTestEmbedder


def test_ingest_then_evaluate_writes_a_versioned_round_trippable_run_artifact(
    tmp_path: Path,
) -> None:
    config_path, dataset_path = _write_fixture_inputs(tmp_path)
    output_path = tmp_path / "runs" / "baseline.json"
    test_embedder = DeterministicTestEmbedder()

    manifest = ingest(config_path, embedder=test_embedder)
    run = evaluate(config_path, dataset_path, output_path, embedder=test_embedder)

    assert manifest.chunk_count > 0
    assert run.schema_version == 1
    assert run.corpus_fingerprint == manifest.corpus_fingerprint
    assert run.dataset_fingerprint
    assert len(run.cases) == 2
    assert run.metrics.recall_at_k == 1.0
    assert run.metrics.hit_rate == 1.0
    assert output_path.exists()
    assert RunArtifact.model_validate_json(output_path.read_text(encoding="utf-8")) == run


def test_evaluate_requires_the_prebuilt_index_bundle(tmp_path: Path) -> None:
    config_path, dataset_path = _write_fixture_inputs(tmp_path)

    with pytest.raises(UserInputError, match="rag-regress ingest"):
        evaluate(
            config_path,
            dataset_path,
            tmp_path / "runs" / "missing.json",
            DeterministicTestEmbedder(),
        )


def test_evaluate_failure_leaves_existing_artifact_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, dataset_path = _write_fixture_inputs(tmp_path)
    test_embedder = DeterministicTestEmbedder()
    ingest(config_path, embedder=test_embedder)
    output_path = tmp_path / "runs" / "interrupted.json"

    original_embed_query = test_embedder.embed_query
    calls = 0

    def fail_second_measured_query(question: str):
        nonlocal calls
        calls += 1
        if calls == 3:  # one warm-up, then the second measured case
            raise RuntimeError("simulated query embedding interruption")
        return original_embed_query(question)

    monkeypatch.setattr(test_embedder, "embed_query", fail_second_measured_query)

    with pytest.raises(RuntimeError, match="simulated query embedding interruption"):
        evaluate(config_path, dataset_path, output_path, embedder=test_embedder)

    assert not output_path.exists()
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []

    original_content = '{"complete":"previous run"}\n'
    output_path.parent.mkdir(parents=True)
    output_path.write_text(original_content, encoding="utf-8")
    calls = 0

    with pytest.raises(RuntimeError, match="simulated query embedding interruption"):
        evaluate(config_path, dataset_path, output_path, embedder=test_embedder)

    assert output_path.read_text(encoding="utf-8") == original_content


def _write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    (corpus_path / "account.md").write_text(
        "password reset account credentials guide", encoding="utf-8"
    )
    (corpus_path / "billing.txt").write_text(
        "invoice billing payment receipt guide", encoding="utf-8"
    )

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "corpus:",
                "  path: corpus",
                "  include:",
                "    - '**/*.md'",
                "    - '**/*.txt'",
                "chunking:",
                "  strategy: words",
                "  size: 20",
                "  overlap: 0",
                "embedding:",
                "  provider: sentence_transformers",
                "  model: offline-test-model",
                "  normalize: true",
                "retrieval:",
                "  metric: cosine",
                "  top_k: 2",
                "  relevance_threshold: null",
                "storage:",
                "  directory: state",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "offline fixture",
                "cases": [
                    {
                        "id": "account",
                        "question": "password reset credentials",
                        "expected_documents": ["account.md"],
                        "expected_text": ["password reset"],
                    },
                    {
                        "id": "billing",
                        "question": "invoice billing payment",
                        "expected_documents": ["billing.txt"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path, dataset_path
