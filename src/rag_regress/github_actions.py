"""GitHub workflow command and compact job-summary presentation."""

import json
import re

from rag_regress.models import CaseComparison, ComparisonReport, GateReport, RunArtifact

_QUALITY_METRICS = ("recall_at_k", "hit_at_k", "reciprocal_rank", "evidence_hit")


def escape_workflow_data(text: str) -> str:
    """Keep data on one workflow-command line, escaping percent first."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _decreases(case: CaseComparison) -> str:
    return "; ".join(
        f"{name} baseline={metric.baseline} candidate={metric.candidate} delta={metric.delta}"
        for name in _QUALITY_METRICS
        if (metric := getattr(case, name)).delta is not None and metric.delta < 0
    )


def render_actions_annotations(
    comparison: ComparisonReport | None,
    gates: GateReport,
) -> tuple[str, ...]:
    """Emit warnings for decreased quality and errors only for failed gates."""
    lines = []
    decreases = (
        [(case, detail) for case in comparison.case_deltas if (detail := _decreases(case))]
        if comparison is not None
        else []
    )
    # GitHub displays at most ten warnings and ten errors per step.
    for case, detail in decreases[:10]:
        lines.append(f"::warning::{escape_workflow_data(f'Case {case.id}: {detail}')}")
    if len(decreases) > 10:
        lines.append(
            f"::notice::{len(decreases) - 10} additional cases have quality decreases; "
            "see the job summary and saved run artifacts."
        )
    for failure in gates.failures[:10]:
        message = f"{failure.gate}: expected {failure.expected}, actual {_value(failure.actual)}"
        if failure.baseline is not None:
            message += f", baseline {failure.baseline}"
        lines.append(f"::error::{escape_workflow_data(message)}")
    if len(gates.failures) > 10:
        lines.append(
            f"::notice::{len(gates.failures) - 10} additional gates failed; see the job summary."
        )
    return tuple(lines)


def render_actions_summary(
    candidate: RunArtifact,
    comparison: ComparisonReport | None,
    gates: GateReport,
) -> str:
    """Summarize metrics and case IDs without publishing questions or passages."""
    lines = ["", "## rag-regress", "", f"Quality gates: {'PASS' if gates.passed else 'FAIL'}", ""]
    if comparison is None:
        lines.extend(["Baseline: not supplied; case decreases were not evaluated.", ""])
    lines.extend(["| Metric | Baseline | Candidate | Delta |", "| --- | ---: | ---: | ---: |"])
    for name in type(candidate.metrics).model_fields:
        metric = getattr(comparison.metrics, name) if comparison is not None else None
        lines.append(
            f"| {name} | {_value(metric.baseline if metric else None)} | "
            f"{_value(getattr(candidate.metrics, name))} | "
            f"{_value(metric.delta if metric else None)} |"
        )
    lines.extend(["", "Deltas are candidate minus baseline; lower latency is better.", ""])
    for failure in gates.failures:
        message = f"{failure.gate}: expected {failure.expected}, actual {_value(failure.actual)}"
        if failure.baseline is not None:
            message += f", baseline {failure.baseline}"
        lines.append(f"- {_literal(message)}")
    if comparison is not None:
        decreases = [
            (case, detail) for case in comparison.case_deltas if (detail := _decreases(case))
        ]
        lines.extend(["", f"Quality decreases: {len(decreases)}", ""])
        for case, detail in decreases[:100]:
            case_id = case.id[:500] + ("... (ID truncated)" if len(case.id) > 500 else "")
            lines.append(f"- {_literal(case_id)}: {detail}")
        if len(decreases) > 100:
            lines.extend(
                [
                    "",
                    f"{len(decreases) - 100} additional cases omitted; "
                    "see saved run artifacts for the complete results.",
                ]
            )
    return "\n".join(lines) + "\n"


def _literal(text: str) -> str:
    text = json.dumps(text, ensure_ascii=True)
    delimiter = "`" * (1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return f"{delimiter} {text} {delimiter}"


def _value(value: float | None) -> str:
    return "n/a" if value is None else str(value)
