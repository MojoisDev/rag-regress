"""Stable command-line presentation for local retrieval regression workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import ValidationError

from rag_regress.comparison import compare_runs
from rag_regress.errors import ExecutionError, UserInputError
from rag_regress.experiments import evaluate, ingest
from rag_regress.models import (
    ComparisonReport,
    GateFailure,
    GateReport,
    IndexBundleManifest,
    MetricComparison,
    RunArtifact,
)
from rag_regress.quality_gates import check_quality_gates, load_quality_gates

app = typer.Typer(
    no_args_is_help=True,
    help="Local-first retrieval regression testing.",
    add_completion=False,
)

_Result = TypeVar("_Result")


@app.callback()
def main(
    ctx: typer.Context,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show a traceback for unexpected errors."),
    ] = False,
) -> None:
    """Configure global presentation options."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@app.command("ingest")
def ingest_command(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Pipeline configuration YAML file."),
    ],
) -> None:
    """Build or reuse the index bundle described by CONFIG."""
    manifest = execute(ctx, lambda: run_ingest(config))
    typer.echo(
        "Index bundle ready: "
        f"{manifest.chunk_count} chunks (corpus {manifest.corpus_fingerprint})"
    )


@app.command("evaluate")
def evaluate_command(
    ctx: typer.Context,
    dataset: Annotated[Path, typer.Argument(help="Evaluation dataset JSON file.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Pipeline configuration YAML file."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination JSON run artifact."),
    ],
) -> None:
    """Evaluate DATASET against a prebuilt bundle and write OUTPUT."""
    run = execute(ctx, lambda: run_evaluate(dataset, config, output))
    typer.echo(f"Metrics: {_format_aggregate_metrics(run)}")
    typer.echo(f"Run artifact: {output}")


@app.command("compare")
def compare_command(
    ctx: typer.Context,
    baseline: Annotated[Path, typer.Argument(help="Baseline JSON run artifact.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate JSON run artifact.")],
) -> None:
    """Compare CANDIDATE with BASELINE."""
    report = execute(ctx, lambda: run_compare(baseline, candidate))
    _print_comparison(report)


@app.command("check")
def check_command(
    ctx: typer.Context,
    candidate: Annotated[Path, typer.Argument(help="Candidate JSON run artifact.")],
    gates: Annotated[
        Path,
        typer.Option("--gates", help="Quality-gate YAML file."),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Baseline JSON run artifact for relative gates."),
    ] = None,
) -> None:
    """Check CANDIDATE against quality GATES."""
    report = execute(ctx, lambda: run_check(candidate, baseline, gates))
    if report.passed:
        typer.echo("PASS: all quality gates passed")
        return

    for failure in report.failures:
        typer.echo(_format_gate_failure(failure))
    raise typer.Exit(1)


def execute(ctx: typer.Context, operation: Callable[[], _Result]) -> _Result:
    """Translate known service failures into the documented user-error exit code."""
    try:
        return operation()
    except (UserInputError, ExecutionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        if _debug_enabled(ctx):
            raise
        typer.echo(
            f"Error: unexpected failure: {exc}. Rerun with --debug for a traceback.",
            err=True,
        )
        raise typer.Exit(2) from exc


def run_ingest(config_path: Path) -> IndexBundleManifest:
    """Call the ingestion application service."""
    return ingest(config_path)


def run_evaluate(dataset_path: Path, config_path: Path, output_path: Path) -> RunArtifact:
    """Call the evaluation application service."""
    return evaluate(config_path, dataset_path, output_path)


def run_compare(baseline_path: Path, candidate_path: Path) -> ComparisonReport:
    """Load two run artifacts and compare them through the domain service."""
    return compare_runs(_load_run_artifact(baseline_path), _load_run_artifact(candidate_path))


def run_check(candidate_path: Path, baseline_path: Path | None, gates_path: Path) -> GateReport:
    """Load the requested artifacts and delegate quality-gate evaluation."""
    baseline = _load_run_artifact(baseline_path) if baseline_path is not None else None
    return check_quality_gates(
        candidate=_load_run_artifact(candidate_path),
        gates=load_quality_gates(gates_path),
        baseline=baseline,
    )


def _load_run_artifact(path: Path) -> RunArtifact:
    """Load one complete JSON run artifact and reject incomplete failure diagnostics."""
    source = path.resolve()
    if source.name.endswith(".failed.json"):
        raise UserInputError(f"Incomplete failed run artifact is not valid here: {source}")
    try:
        return RunArtifact.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise UserInputError(f"Invalid run artifact {source}: {exc}") from exc


def _debug_enabled(ctx: typer.Context) -> bool:
    return isinstance(ctx.obj, dict) and bool(ctx.obj.get("debug"))


def _format_aggregate_metrics(run: RunArtifact) -> str:
    metrics = run.metrics
    values = (
        ("recall_at_k", metrics.recall_at_k),
        ("hit_rate", metrics.hit_rate),
        ("mrr", metrics.mrr),
        ("evidence_hit_rate", metrics.evidence_hit_rate),
        ("p50_retrieval_ms", metrics.p50_retrieval_ms),
        ("p95_retrieval_ms", metrics.p95_retrieval_ms),
    )
    return " ".join(f"{name}={_format_value(value)}" for name, value in values)


def _print_comparison(report: ComparisonReport) -> None:
    for name, comparison in (
        ("recall_at_k", report.metrics.recall_at_k),
        ("hit_rate", report.metrics.hit_rate),
        ("mrr", report.metrics.mrr),
        ("evidence_hit_rate", report.metrics.evidence_hit_rate),
        ("p50_retrieval_ms", report.metrics.p50_retrieval_ms),
        ("p95_retrieval_ms", report.metrics.p95_retrieval_ms),
    ):
        typer.echo(f"{name}: {_format_metric_comparison(comparison)}")
    typer.echo(f"Improved: {_format_case_ids(report.improved_case_ids)}")
    typer.echo(f"Regressed: {_format_case_ids(report.regressed_case_ids)}")
    typer.echo(f"Unchanged: {_format_case_ids(report.unchanged_case_ids)}")


def _format_metric_comparison(comparison: MetricComparison) -> str:
    return (
        f"baseline={_format_value(comparison.baseline)} "
        f"candidate={_format_value(comparison.candidate)} "
        f"delta={_format_value(comparison.delta)}"
    )


def _format_gate_failure(failure: GateFailure) -> str:
    result = (
        f"FAIL: {failure.gate}: expected {failure.expected}, "
        f"actual {_format_value(failure.actual)}"
    )
    if failure.baseline is not None:
        result += f", baseline {_format_value(failure.baseline)}"
    return result


def _format_case_ids(case_ids: tuple[str, ...]) -> str:
    return ", ".join(case_ids) if case_ids else "(none)"


def _format_value(value: float | None) -> str:
    return "n/a" if value is None else str(value)
