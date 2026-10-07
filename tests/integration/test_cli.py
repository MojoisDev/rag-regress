from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rag_regress import cli
from rag_regress.errors import UserInputError
from rag_regress.models import (
    AggregateMetricComparison,
    ComparisonReport,
    GateFailure,
    GateReport,
    MetricComparison,
)

runner = CliRunner()


def test_check_returns_one_after_printing_each_failed_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A false gate report must be a CI failure, not an input error."""
    report = GateReport(
        passed=False,
        failures=(
            GateFailure(gate="minimum.mrr", expected=">= 0.8", actual=0.7),
            GateFailure(
                gate="maximum.p95_retrieval_ms",
                expected="<= 100.0",
                actual=120.0,
            ),
        ),
    )
    monkeypatch.setattr(cli, "run_check", lambda *args, **kwargs: report)

    result = runner.invoke(
        cli.app,
        ["check", "candidate.json", "--baseline", "baseline.json", "--gates", "gates.yaml"],
    )

    assert result.exit_code == 1
    assert "FAIL: minimum.mrr" in result.stdout
    assert "FAIL: maximum.p95_retrieval_ms" in result.stdout


def test_evaluate_user_error_is_written_to_stderr_with_exit_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad user input must remain concise and distinguishable from gate failures."""

    def fail(*args: object, **kwargs: object) -> None:
        raise UserInputError("bad dataset")

    monkeypatch.setattr(cli, "run_evaluate", fail)

    result = runner.invoke(
        cli.app,
        ["evaluate", "eval.json", "--config", "config.yaml", "--output", "run.json"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error: bad dataset" in result.stderr
    assert "Traceback" not in result.stderr


def test_debug_mode_preserves_an_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing debug re-raising would hide the exception's traceback from operators."""

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected model state")

    monkeypatch.setattr(cli, "run_evaluate", fail)

    result = runner.invoke(
        cli.app,
        ["--debug", "evaluate", "eval.json", "--config", "config.yaml", "--output", "run.json"],
    )

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "unexpected model state"


def test_compare_prints_metric_deltas_and_case_classifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison result must give humans and CI its aggregate and case summary."""
    report = ComparisonReport(
        metrics=AggregateMetricComparison(
            recall_at_k=MetricComparison(baseline=0.8, candidate=0.9, delta=0.1),
            hit_rate=MetricComparison(baseline=0.8, candidate=0.8, delta=0.0),
            mrr=MetricComparison(baseline=0.8, candidate=0.7, delta=-0.1),
            evidence_hit_rate=MetricComparison(baseline=None, candidate=None, delta=None),
            p50_retrieval_ms=MetricComparison(baseline=10.0, candidate=8.0, delta=-2.0),
            p95_retrieval_ms=MetricComparison(baseline=20.0, candidate=25.0, delta=5.0),
        ),
        case_deltas=(),
        improved_case_ids=("better",),
        regressed_case_ids=("worse",),
        unchanged_case_ids=("same",),
    )
    monkeypatch.setattr(cli, "run_compare", lambda *args, **kwargs: report)

    result = runner.invoke(cli.app, ["compare", "baseline.json", "candidate.json"])

    assert result.exit_code == 0
    assert "recall_at_k: baseline=0.8 candidate=0.9 delta=0.1" in result.stdout
    assert "Improved: better" in result.stdout
    assert "Regressed: worse" in result.stdout
    assert "Unchanged: same" in result.stdout


def test_successful_ingest_and_evaluate_report_their_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success output must tell an operator what was built and where a run was written."""
    monkeypatch.setattr(
        cli,
        "run_ingest",
        lambda *args, **kwargs: SimpleNamespace(chunk_count=4, corpus_fingerprint="corpus-123"),
    )
    monkeypatch.setattr(
        cli,
        "run_evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            metrics=SimpleNamespace(
                recall_at_k=1.0,
                hit_rate=1.0,
                mrr=1.0,
                evidence_hit_rate=None,
                p50_retrieval_ms=3.0,
                p95_retrieval_ms=5.0,
            )
        ),
    )

    ingest_result = runner.invoke(cli.app, ["ingest", "--config", "pipeline.yaml"])
    evaluate_result = runner.invoke(
        cli.app,
        ["evaluate", "eval.json", "--config", "pipeline.yaml", "--output", "run.json"],
    )

    assert ingest_result.exit_code == 0
    assert "Index bundle ready: 4 chunks (corpus corpus-123)" in ingest_result.stdout
    assert evaluate_result.exit_code == 0
    assert "mrr=1.0" in evaluate_result.stdout
    assert "Run artifact: run.json" in evaluate_result.stdout


@pytest.mark.parametrize(
    ("arguments", "required_terms"),
    [
        (["ingest", "--help"], ("--config",)),
        (["evaluate", "--help"], ("DATASET", "--config", "--output")),
        (["compare", "--help"], ("BASELINE", "CANDIDATE")),
        (["check", "--help"], ("CANDIDATE", "--baseline", "--gates")),
    ],
)
def test_command_help_lists_the_stable_documented_arguments(
    arguments: list[str], required_terms: tuple[str, ...]
) -> None:
    """Changing a documented command argument should fail this public CLI contract test."""
    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 0
    for term in required_terms:
        assert term in result.stdout
