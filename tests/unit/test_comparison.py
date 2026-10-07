from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_regress.comparison import compare_runs
from rag_regress.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from rag_regress.embeddings import EmbeddingMetadata
from rag_regress.errors import UserInputError
from rag_regress.models import (
    AggregateMetrics,
    CaseMetrics,
    CaseResult,
    EnvironmentInfo,
    PipelineSnapshot,
    RunArtifact,
)


def test_incompatible_runs_list_every_compatibility_mismatch() -> None:
    baseline = make_run(
        schema_version=1,
        corpus_fingerprint="corpus-a",
        dataset_fingerprint="dataset-a",
        metric_definition_version="1",
        case_ids=("first", "second"),
    )
    candidate = make_run(
        schema_version=2,
        corpus_fingerprint="corpus-b",
        dataset_fingerprint="dataset-b",
        metric_definition_version="2",
        case_ids=("second", "first"),
    )

    with pytest.raises(UserInputError) as captured:
        compare_runs(baseline, candidate)

    message = str(captured.value)
    for field in (
        "schema_version",
        "corpus_fingerprint",
        "dataset_fingerprint",
        "case_ids",
        "metric_definition_version",
    ):
        assert field in message


def test_cases_are_classified_by_reciprocal_rank_delta_in_stable_order() -> None:
    baseline = make_run_with_reciprocal_ranks({"same": 0.0, "worse": 1.0, "better": 0.5})
    candidate = make_run_with_reciprocal_ranks({"same": 0.0, "worse": 0.5, "better": 1.0})

    report = compare_runs(baseline, candidate)

    assert report.improved_case_ids == ("better",)
    assert report.regressed_case_ids == ("worse",)
    assert report.unchanged_case_ids == ("same",)
    assert [delta.id for delta in report.case_deltas] == ["better", "same", "worse"]


def test_comparison_reports_candidate_minus_baseline_aggregate_deltas() -> None:
    baseline = make_run(recall=0.7, hit_rate=0.8, mrr=0.4, p50=10, p95=20)
    candidate = make_run(recall=0.9, hit_rate=0.6, mrr=0.5, p50=7, p95=25)

    report = compare_runs(baseline, candidate)

    assert report.metrics.recall_at_k.baseline == 0.7
    assert report.metrics.recall_at_k.candidate == 0.9
    assert report.metrics.recall_at_k.delta == pytest.approx(0.2)
    assert report.metrics.p50_retrieval_ms.delta == -3
    assert report.metrics.p95_retrieval_ms.delta == 5


def test_comparison_preserves_nullable_evidence_metric_when_not_measured() -> None:
    report = compare_runs(make_run(evidence=None), make_run(evidence=None))

    assert report.metrics.evidence_hit_rate.baseline is None
    assert report.metrics.evidence_hit_rate.candidate is None
    assert report.metrics.evidence_hit_rate.delta is None


def make_run_with_reciprocal_ranks(values: dict[str, float]) -> RunArtifact:
    return make_run(
        case_ids=tuple(values),
        case_reciprocal_ranks=values,
    )


def make_run(
    *,
    schema_version: int = 1,
    corpus_fingerprint: str = "corpus",
    dataset_fingerprint: str = "dataset",
    metric_definition_version: str = "1",
    case_ids: tuple[str, ...] = ("case",),
    case_reciprocal_ranks: dict[str, float] | None = None,
    recall: float = 1.0,
    hit_rate: float = 1.0,
    mrr: float = 1.0,
    evidence: float | None = None,
    p50: float = 10.0,
    p95: float = 20.0,
) -> RunArtifact:
    ranks = case_reciprocal_ranks or {case_id: 1.0 for case_id in case_ids}
    return RunArtifact.model_construct(
        schema_version=schema_version,
        tool_version="0.1.0",
        metric_definition_version=metric_definition_version,
        dataset_name="test dataset",
        corpus_fingerprint=corpus_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        pipeline=PipelineSnapshot(
            corpus_include_patterns=("**/*.md",),
            chunking=ChunkingConfig(strategy="words", size=10, overlap=0),
            embedding=EmbeddingConfig(
                provider="sentence_transformers", model="test", normalize=True
            ),
            retrieval=RetrievalConfig(metric="cosine", top_k=1, relevance_threshold=None),
        ),
        environment=EnvironmentInfo(
            python_version="3.11",
            platform="test",
            numpy_version="1",
            faiss_version="1",
            sentence_transformers_version="1",
            rag_regress_version="0.1.0",
        ),
        embedding=EmbeddingMetadata(provider="sentence_transformers", model="test"),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        warnings=(),
        errors=(),
        metrics=AggregateMetrics(
            recall_at_k=recall,
            hit_rate=hit_rate,
            mrr=mrr,
            evidence_hit_rate=evidence,
            p50_retrieval_ms=p50,
            p95_retrieval_ms=p95,
        ),
        cases=tuple(
            CaseResult(
                id=case_id,
                question=f"Question for {case_id}",
                expected_documents=("doc.md",),
                expected_text=(),
                results=(),
                metrics=CaseMetrics(
                    recall_at_k=ranks[case_id],
                    hit_at_k=ranks[case_id],
                    reciprocal_rank=ranks[case_id],
                    evidence_hit=None,
                ),
                retrieval_ms=1.0,
            )
            for case_id in case_ids
        ),
    )
