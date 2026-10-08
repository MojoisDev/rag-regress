from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_regress.embeddings import EmbeddingRuntime
from rag_regress.errors import UserInputError
from rag_regress.experiments import create_run_artifact, evaluate, ingest
from rag_regress.models import (
    CaseMetrics,
    CaseResult,
    CorpusSnapshot,
    Document,
    EvaluationCase,
    EvaluationDataset,
    RetrievalResult,
    RunArtifact,
)
from rag_regress.quality_gates import QualityGateConfig, check_quality_gates
from tests.helpers import DeterministicTestEmbedder


@pytest.mark.parametrize("device,gpu_name", [("cpu", None), ("cuda:0", "Test GPU")])
def test_evaluation_records_observed_runtime_and_reads_legacy_artifacts(
    tmp_path: Path,
    device: str,
    gpu_name: str | None,
) -> None:
    class RuntimeEmbedder(DeterministicTestEmbedder):
        @property
        def runtime(self) -> EmbeddingRuntime:
            return EmbeddingRuntime(device, gpu_name, "test-torch", "test-cuda")

    config_path, dataset_path = _write_fixture_inputs(tmp_path)
    embedder = RuntimeEmbedder()
    ingest(config_path, embedder)
    output = tmp_path / "run.json"
    run = evaluate(config_path, dataset_path, output, embedder)
    payload = json.loads(output.read_text())
    expected = {
        "embedding_device": device,
        "gpu_name": gpu_name,
        "torch_version": "test-torch",
        "cuda_version": "test-cuda",
    }
    for key, value in expected.items():
        assert payload["environment"][key] == value
        assert getattr(run.environment, key) == value
    assert RunArtifact.model_validate(payload) == run
    for key in expected:
        del payload["environment"][key]
    payload["pipeline"]["embedding"].pop("device", None)
    legacy = RunArtifact.model_validate(payload)
    assert legacy.environment.embedding_device is None
    assert legacy.pipeline.embedding.device == "auto"


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
    assert run.embedding.revision == manifest.embedding.revision == "a" * 40
    assert run.pipeline.embedding.revision == "a" * 40
    assert run.environment.embedding_device is None
    assert run.environment.torch_version is None
    assert run.dataset_fingerprint
    assert len(run.cases) == 2
    assert run.metrics.recall_at_k == 1.0
    assert run.metrics.hit_rate == 1.0
    assert output_path.exists()
    assert RunArtifact.model_validate_json(output_path.read_text(encoding="utf-8")) == run


def test_repeated_latency_excludes_full_warmups_and_aggregates_all_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_regress import experiments
    from rag_regress.index import FaissIndex

    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(config.read_text() + "latency: {warmup_queries: 3, repetitions: 3}\n")
    embedder = DeterministicTestEmbedder()
    ingest(config, embedder)
    questions: list[str] = []
    searches = 0
    original_query = embedder.embed_query
    original_search = FaissIndex.search

    def record_query(question):
        questions.append(question)
        return original_query(question)

    def record_search(self, query, top_k):
        nonlocal searches
        searches += 1
        return original_search(self, query, top_k)

    monkeypatch.setattr(embedder, "embed_query", record_query)
    monkeypatch.setattr(FaissIndex, "search", record_search)
    # Three warmups, the run timer, six measured retrievals, then the run duration.
    ticks = iter([0, 1, 2, 3, 4, 5, 6, 10, 11, 20, 23, 30, 32, 40, 44, 50, 59, 60, 160, 161])
    monkeypatch.setattr(experiments.time, "perf_counter", lambda: next(ticks))
    run = evaluate(config, dataset, tmp_path / "run.json", embedder)

    assert (
        questions
        == ["password reset credentials", "invoice billing payment", "password reset credentials"]
        + ["password reset credentials"] * 3
        + ["invoice billing payment"] * 3
    )
    assert searches == 9
    assert [case.retrieval_samples_ms for case in run.cases] == [
        (1000.0, 3000.0, 2000.0),
        (4000.0, 9000.0, 100000.0),
    ]
    assert [case.retrieval_ms for case in run.cases] == [2000.0, 9000.0]
    assert run.metrics.p50_retrieval_ms == 3000.0
    assert run.metrics.p95_retrieval_ms == 100000.0
    assert len(run.cases) == 2 and run.metrics.recall_at_k == 1.0
    assert run.pipeline.latency.warmup_queries == 3
    assert run.pipeline.latency.repetitions == 3


def test_latency_defaults_zero_warmup_and_legacy_artifacts(tmp_path: Path) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    embedder = DeterministicTestEmbedder()
    ingest(config, embedder)
    run = evaluate(config, dataset, tmp_path / "default.json", embedder)
    assert all(case.retrieval_samples_ms == (case.retrieval_ms,) for case in run.cases)
    payload = run.model_dump(mode="json")
    del payload["pipeline"]["latency"]
    for case in payload["cases"]:
        del case["retrieval_samples_ms"]
    legacy = RunArtifact.model_validate(payload)
    assert legacy.pipeline.latency.repetitions == 1
    assert all(case.retrieval_samples_ms == () for case in legacy.cases)

    config.write_text(config.read_text() + "latency: {warmup_queries: 0, repetitions: 2}\n")

    class CountingEmbedder(DeterministicTestEmbedder):
        calls = 0

        def embed_query(self, question):
            self.calls += 1
            return super().embed_query(question)

    counted = CountingEmbedder()
    measured = evaluate(config, dataset, tmp_path / "zero.json", counted)
    assert counted.calls == 4
    assert all(len(case.retrieval_samples_ms) == 2 for case in measured.cases)


def test_evaluate_requires_the_prebuilt_index_bundle(tmp_path: Path) -> None:
    config_path, dataset_path = _write_fixture_inputs(tmp_path)

    with pytest.raises(UserInputError, match="rag-regress ingest"):
        evaluate(
            config_path,
            dataset_path,
            tmp_path / "runs" / "missing.json",
            DeterministicTestEmbedder(),
        )


@pytest.mark.parametrize("settings", ["", "latency: {warmup_queries: 0, repetitions: 3}\n"])
def test_evaluate_failure_leaves_existing_artifact_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: str
) -> None:
    config_path, dataset_path = _write_fixture_inputs(tmp_path)
    config_path.write_text(config_path.read_text() + settings)
    test_embedder = DeterministicTestEmbedder()
    ingest(config_path, embedder=test_embedder)
    output_path = tmp_path / "runs" / "interrupted.json"

    original_embed_query = test_embedder.embed_query
    calls = 0

    def fail_second_measured_query(question: str):
        nonlocal calls
        calls += 1
        if calls == 3:  # second case by default, or third repetition with zero warmup
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


def test_repetitions_keep_first_results_for_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(config.read_text() + "latency: {warmup_queries: 0, repetitions: 3}\n")
    embedder = DeterministicTestEmbedder()
    ingest(config, embedder)
    original_query = embedder.embed_query
    calls = 0

    def varied_query(question):
        nonlocal calls
        calls += 1
        # Make subsequent measured queries different; only the first determines quality.
        opposite = (
            "invoice billing payment"
            if question == "password reset credentials"
            else "password reset credentials"
        )
        return original_query(question if calls % 3 == 1 else opposite)

    monkeypatch.setattr(embedder, "embed_query", varied_query)
    run = evaluate(config, dataset, tmp_path / "run.json", embedder)
    assert calls == 6
    assert all(case.results[0].document_path == case.expected_documents[0] for case in run.cases)
    assert run.metrics.mrr == run.metrics.hit_rate == run.metrics.recall_at_k == 1.0


def test_exact_rank_aggregation_obeys_inclusive_mrr_gate(tmp_path: Path) -> None:
    config_path, _ = _write_fixture_inputs(tmp_path)
    from rag_regress.config import load_pipeline_config

    config = load_pipeline_config(config_path)
    exact_candidate = _run_for_relevant_ranks(config, (3, 15))
    lower_candidate = _run_for_relevant_ranks(config, (4, 15))
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "minimum": {"mrr": 0.2}}
    )

    assert exact_candidate.metrics.mrr == 0.2
    assert exact_candidate.metrics.p50_retrieval_ms == 1.0
    assert exact_candidate.metrics.p95_retrieval_ms == 1.0
    assert check_quality_gates(exact_candidate, gates).passed is True
    assert check_quality_gates(lower_candidate, gates).passed is False


def _run_for_relevant_ranks(config, ranks: tuple[int, ...]) -> RunArtifact:
    document = Document(
        id="document",
        relative_path="guide.md",
        title="guide",
        content="Guide",
        checksum="checksum",
    )
    corpus = CorpusSnapshot(documents=(document,), fingerprint="corpus")
    dataset = EvaluationDataset(
        schema_version=1,
        name="rank boundaries",
        cases=tuple(
            EvaluationCase(
                id=f"case-{index}",
                question=f"Question {index}?",
                expected_documents=("guide.md",),
            )
            for index in range(len(ranks))
        ),
    )
    cases = tuple(
        CaseResult(
            id=dataset_case.id,
            question=dataset_case.question,
            expected_documents=dataset_case.expected_documents,
            expected_text=(),
            results=tuple(
                RetrievalResult(
                    chunk_id=f"chunk-{result_rank}",
                    document_id=(
                        document.id if result_rank == rank else "irrelevant-document"
                    ),
                    document_path=(
                        document.relative_path if result_rank == rank else "irrelevant.md"
                    ),
                    rank=result_rank,
                    score=0.5,
                    text=document.content if result_rank == rank else "Irrelevant",
                )
                for result_rank in range(1, rank + 1)
            ),
            metrics=CaseMetrics(
                recall_at_k=1.0,
                hit_at_k=1.0,
                reciprocal_rank=1.0 / rank,
                evidence_hit=None,
            ),
            retrieval_ms=1.0,
        )
        for dataset_case, rank in zip(dataset.cases, ranks, strict=True)
    )
    return create_run_artifact(
        config=config,
        corpus=corpus,
        dataset=dataset,
        embedding=DeterministicTestEmbedder().metadata,
        cases=cases,
        started_at=datetime.now(UTC),
        duration_ms=1.0,
    )


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
                "  revision: " + "a" * 40,
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
