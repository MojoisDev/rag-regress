"""Pure comparisons for compatible retrieval evaluation runs."""

from rag_regress.errors import UserInputError
from rag_regress.models import (
    AggregateMetricComparison,
    AggregateMetrics,
    CaseComparison,
    CaseResult,
    ComparisonReport,
    MetricComparison,
    RunArtifact,
)


def compare_runs(baseline: RunArtifact, candidate: RunArtifact) -> ComparisonReport:
    """Compare a candidate run with a compatible baseline run."""
    mismatches = _compatibility_mismatches(baseline, candidate)
    if mismatches:
        raise UserInputError("Incompatible runs: " + ", ".join(mismatches))

    baseline_cases = {case.id: case for case in baseline.cases}
    case_deltas = tuple(
        _compare_case(baseline_cases[case.id], case)
        for case in sorted(candidate.cases, key=lambda item: item.id)
    )
    return _build_comparison_report(baseline.metrics, candidate.metrics, case_deltas)


def _compatibility_mismatches(baseline: RunArtifact, candidate: RunArtifact) -> list[str]:
    comparisons = (
        ("schema_version", baseline.schema_version, candidate.schema_version),
        ("corpus_fingerprint", baseline.corpus_fingerprint, candidate.corpus_fingerprint),
        ("dataset_fingerprint", baseline.dataset_fingerprint, candidate.dataset_fingerprint),
        (
            "case_ids",
            tuple(case.id for case in baseline.cases),
            tuple(case.id for case in candidate.cases),
        ),
        (
            "metric_definition_version",
            baseline.metric_definition_version,
            candidate.metric_definition_version,
        ),
    )
    return [
        name
        for name, baseline_value, candidate_value in comparisons
        if baseline_value != candidate_value
    ]


def _compare_case(baseline: CaseResult, candidate: CaseResult) -> CaseComparison:
    return CaseComparison(
        id=candidate.id,
        recall_at_k=_metric_comparison(
            baseline.metrics.recall_at_k, candidate.metrics.recall_at_k
        ),
        hit_at_k=_metric_comparison(baseline.metrics.hit_at_k, candidate.metrics.hit_at_k),
        reciprocal_rank=_metric_comparison(
            baseline.metrics.reciprocal_rank, candidate.metrics.reciprocal_rank
        ),
        evidence_hit=_metric_comparison(
            baseline.metrics.evidence_hit, candidate.metrics.evidence_hit
        ),
        retrieval_ms=_metric_comparison(baseline.retrieval_ms, candidate.retrieval_ms),
    )


def _build_comparison_report(
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
    case_deltas: tuple[CaseComparison, ...],
) -> ComparisonReport:
    improved = tuple(
        case.id
        for case in case_deltas
        if case.reciprocal_rank.delta is not None and case.reciprocal_rank.delta > 0
    )
    regressed = tuple(
        case.id
        for case in case_deltas
        if case.reciprocal_rank.delta is not None and case.reciprocal_rank.delta < 0
    )
    unchanged = tuple(
        case.id
        for case in case_deltas
        if case.reciprocal_rank.delta == 0
    )
    return ComparisonReport(
        metrics=AggregateMetricComparison(
            recall_at_k=_metric_comparison(baseline.recall_at_k, candidate.recall_at_k),
            hit_rate=_metric_comparison(baseline.hit_rate, candidate.hit_rate),
            mrr=_metric_comparison(baseline.mrr, candidate.mrr),
            evidence_hit_rate=_metric_comparison(
                baseline.evidence_hit_rate, candidate.evidence_hit_rate
            ),
            p50_retrieval_ms=_metric_comparison(
                baseline.p50_retrieval_ms, candidate.p50_retrieval_ms
            ),
            p95_retrieval_ms=_metric_comparison(
                baseline.p95_retrieval_ms, candidate.p95_retrieval_ms
            ),
        ),
        case_deltas=case_deltas,
        improved_case_ids=improved,
        regressed_case_ids=regressed,
        unchanged_case_ids=unchanged,
    )


def _metric_comparison(baseline: float | None, candidate: float | None) -> MetricComparison:
    """Build a delta only when both values are defined."""
    delta = candidate - baseline if baseline is not None and candidate is not None else None
    return MetricComparison(baseline=baseline, candidate=candidate, delta=delta)
