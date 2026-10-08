"""Local Markdown presentation for compatible retrieval runs."""

import json

from rag_regress.comparison import compare_runs
from rag_regress.models import CaseResult, GateReport, MetricComparison, RunArtifact


def render_comparison_markdown(
    baseline: RunArtifact,
    candidate: RunArtifact,
    gates: GateReport | None = None,
) -> str:
    """Render saved runs literally, without loading a corpus or embedding model."""
    comparison = compare_runs(baseline, candidate)
    lines = ["# Retrieval comparison", "", "## Run context", ""]
    for label, run in (("Baseline", baseline), ("Candidate", candidate)):
        lines.extend(
            [
                f"### {label}",
                "",
                _literal(
                    json.dumps(
                        {
                            "dataset": run.dataset_name,
                            "corpus_fingerprint": run.corpus_fingerprint,
                            "dataset_fingerprint": run.dataset_fingerprint,
                            "pipeline": run.pipeline.model_dump(mode="json"),
                            "environment": run.environment.model_dump(mode="json"),
                            "warnings": run.warnings,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                ),
                "",
            ]
        )
    baseline_settings = baseline.pipeline.model_dump().get("latency")
    candidate_settings = candidate.pipeline.model_dump().get("latency")
    if "latency" not in baseline.pipeline.model_fields_set or (
        "latency" not in candidate.pipeline.model_fields_set
    ):
        lines.extend(
            [
                "Legacy run: latency settings were not recorded; "
                "per-case latency is a single measurement.",
                "",
            ]
        )
    if baseline_settings != candidate_settings:
        lines.extend(
            ["Latency measurement settings differ; interpret latency deltas with care.", ""]
        )
    lines.extend(
        [
            "## Aggregate metrics",
            "",
            _metric_table(
                {
                    name: getattr(comparison.metrics, name)
                    for name in type(comparison.metrics).model_fields
                }
            ),
            "",
            "Deltas are candidate minus baseline. Higher quality and lower latency are better.",
            "",
            "## Case classification",
            "",
            "Classification uses reciprocal rank only.",
            "",
        ]
    )
    for label, ids in (
        ("Improved", comparison.improved_case_ids),
        ("Regressed", comparison.regressed_case_ids),
        ("Unchanged", comparison.unchanged_case_ids),
    ):
        lines.extend([f"{label}: {len(ids)}", "", _literal("\n".join(ids) or "(none)"), ""])
    if gates is None:
        lines.extend(["Quality gates: not supplied", ""])
    else:
        lines.extend(
            ["## Quality gates", "", f"Quality gates: {'PASS' if gates.passed else 'FAIL'}", ""]
        )
        for failure in gates.failures:
            lines.extend(
                [
                    _literal(
                        f"{failure.gate}: expected {failure.expected}, "
                        f"actual {_value(failure.actual)}"
                        + (
                            f", baseline {_value(failure.baseline)}"
                            if failure.baseline is not None
                            else ""
                        )
                    ),
                    "",
                ]
            )
    quality_names = ("recall_at_k", "hit_at_k", "reciprocal_rank", "evidence_hit")
    decreases = [
        case
        for case in comparison.case_deltas
        if any(
            getattr(case, name).delta is not None and getattr(case, name).delta < 0
            for name in quality_names
        )
    ]
    lines.extend(
        [
            "## Questions with quality decreases",
            "",
            f"Questions with quality decreases: {len(decreases)}",
            "",
        ]
    )
    if not decreases:
        lines.extend(["No questions have decreased quality metrics.", ""])
    baseline_cases = {case.id: case for case in baseline.cases}
    candidate_cases = {case.id: case for case in candidate.cases}
    for position, delta in enumerate(decreases, start=1):
        case = candidate_cases[delta.id]
        lines.extend(
            [
                f"### Question {position}",
                "",
                "Case ID:",
                "",
                _literal(case.id),
                "",
                "Question:",
                "",
                _literal(case.question),
                "",
                "Expected documents:",
                "",
                _literal("\n".join(case.expected_documents)),
                "",
                "Expected evidence:",
                "",
                _literal("\n".join(case.expected_text) or "(none)"),
                "",
                _metric_table({name: getattr(delta, name) for name in quality_names}),
                "",
            ]
        )
        for label, result in (("Baseline", baseline_cases[delta.id]), ("Candidate", case)):
            latency_label = (
                "p50 retrieval ms" if result.retrieval_samples_ms else "retrieval ms (single)"
            )
            lines.extend(
                [f"#### {label} passages", "", f"{latency_label}: {result.retrieval_ms}", ""]
            )
            lines.extend(_passages(result))
    return "\n".join(lines)


def _metric_table(metrics: dict[str, MetricComparison]) -> str:
    lines = ["| Metric | Baseline | Candidate | Delta |", "| --- | ---: | ---: | ---: |"]
    lines.extend(
        f"| {name} | {_value(metric.baseline)} | {_value(metric.candidate)} | "
        f"{_value(metric.delta)} |"
        for name, metric in metrics.items()
    )
    return "\n".join(lines)


def _value(value: float | None) -> str:
    return "n/a" if value is None else str(value)


def _literal(text: str) -> str:
    # Indented blocks keep corpus text literal even when it contains Markdown fences.
    return "\n".join("    " + line for line in text.splitlines())


def _passages(case: CaseResult) -> list[str]:
    if not case.results:
        return ["No retrieved passages.", ""]
    lines = []
    for result in case.results:
        lines.extend(
            [
                f"Rank: {result.rank}; Score: {result.score}",
                "",
                "Document path:",
                "",
                _literal(result.document_path),
                "",
                "Chunk ID:",
                "",
                _literal(result.chunk_id),
                "",
                "Passage:",
                "",
                _literal(result.text),
                "",
            ]
        )
    return lines
