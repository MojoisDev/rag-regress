from __future__ import annotations

import json
from pathlib import Path
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
from tests.unit.test_comparison import make_run

runner = CliRunner()


def test_check_returns_one_after_printing_each_failed_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch.chdir(tmp_path)
    _write_input_files("candidate.json", "baseline.json", "gates.yaml")

    result = runner.invoke(
        cli.app,
        ["check", "candidate.json", "--baseline", "baseline.json", "--gates", "gates.yaml"],
    )

    assert result.exit_code == 1
    assert "FAIL: minimum.mrr" in result.stdout
    assert "FAIL: maximum.p95_retrieval_ms" in result.stdout


def test_evaluate_user_error_is_written_to_stderr_with_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bad user input must remain concise and distinguishable from gate failures."""

    def fail(*args: object, **kwargs: object) -> None:
        raise UserInputError("bad dataset")

    monkeypatch.setattr(cli, "run_evaluate", fail)
    monkeypatch.chdir(tmp_path)
    _write_input_files("eval.json", "config.yaml")

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
    tmp_path: Path,
) -> None:
    """Removing debug re-raising would hide the exception's traceback from operators."""

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected model state")

    monkeypatch.setattr(cli, "run_evaluate", fail)
    monkeypatch.chdir(tmp_path)
    _write_input_files("eval.json", "config.yaml")

    result = runner.invoke(
        cli.app,
        ["--debug", "evaluate", "eval.json", "--config", "config.yaml", "--output", "run.json"],
    )

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "unexpected model state"


def test_compare_prints_metric_deltas_and_case_classifications(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch.chdir(tmp_path)
    _write_input_files("baseline.json", "candidate.json")

    result = runner.invoke(cli.app, ["compare", "baseline.json", "candidate.json"])

    assert result.exit_code == 0
    assert "recall_at_k: baseline=0.8 candidate=0.9 delta=0.1" in result.stdout
    assert "Improved: better" in result.stdout
    assert "Regressed: worse" in result.stdout
    assert "Unchanged: same" in result.stdout


def test_successful_ingest_and_evaluate_report_their_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch.chdir(tmp_path)
    _write_input_files("pipeline.yaml", "eval.json")

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


@pytest.mark.parametrize("missing_field", ["schema_version", "metric_definition_version"])
def test_compare_rejects_run_artifacts_with_omitted_version_fields(
    tmp_path: Path, missing_field: str
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_payload = make_run().model_dump(mode="json")
    candidate_payload = make_run().model_dump(mode="json")
    del candidate_payload[missing_field]
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = runner.invoke(cli.app, ["compare", str(baseline_path), str(candidate_path)])

    assert result.exit_code == 2
    assert f"Invalid run artifact {candidate_path}" in result.stderr
    assert missing_field in result.stderr


def test_compare_rejects_non_finite_numbers_in_run_json(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_payload = make_run().model_dump(mode="json")
    candidate_payload = make_run().model_dump(mode="json")
    candidate_payload["metrics"]["p95_retrieval_ms"] = float("inf")
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = runner.invoke(cli.app, ["compare", str(baseline_path), str(candidate_path)])

    assert result.exit_code == 2
    assert f"Invalid run artifact {candidate_path}" in result.stderr
    assert "p95_retrieval_ms" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "valid_files", "boundary"),
    [
        (["ingest", "--config", "missing.yaml"], (), "run_ingest"),
        (
            ["evaluate", "missing.json", "--config", "config.yaml", "--output", "run.json"],
            ("config.yaml",),
            "run_evaluate",
        ),
        (
            ["evaluate", "dataset.json", "--config", "missing.yaml", "--output", "run.json"],
            ("dataset.json",),
            "run_evaluate",
        ),
        (["compare", "missing.json", "candidate.json"], ("candidate.json",), "run_compare"),
        (["compare", "baseline.json", "missing.json"], ("baseline.json",), "run_compare"),
        (
            ["check", "missing.json", "--gates", "gates.yaml"],
            ("gates.yaml",),
            "run_check",
        ),
        (
            ["check", "candidate.json", "--baseline", "missing.json", "--gates", "gates.yaml"],
            ("candidate.json", "gates.yaml"),
            "run_check",
        ),
        (
            ["check", "candidate.json", "--gates", "missing.yaml"],
            ("candidate.json",),
            "run_check",
        ),
    ],
)
def test_commands_reject_missing_read_input_before_calling_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    valid_files: tuple[str, ...],
    boundary: str,
) -> None:
    """A missing read path must be a CLI usage error, never a service invocation."""
    monkeypatch.chdir(tmp_path)
    _write_input_files(*valid_files)
    monkeypatch.setattr(cli, boundary, _service_must_not_run)

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 2
    assert "does not exist" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "valid_files", "directory_name", "boundary"),
    [
        (["ingest", "--config", "directory"], (), "directory", "run_ingest"),
        (
            ["evaluate", "directory", "--config", "config.yaml", "--output", "run.json"],
            ("config.yaml",),
            "directory",
            "run_evaluate",
        ),
        (
            ["evaluate", "dataset.json", "--config", "directory", "--output", "run.json"],
            ("dataset.json",),
            "directory",
            "run_evaluate",
        ),
        (
            ["compare", "directory", "candidate.json"],
            ("candidate.json",),
            "directory",
            "run_compare",
        ),
        (["compare", "baseline.json", "directory"], ("baseline.json",), "directory", "run_compare"),
        (
            ["check", "directory", "--gates", "gates.yaml"],
            ("gates.yaml",),
            "directory",
            "run_check",
        ),
        (
            ["check", "candidate.json", "--baseline", "directory", "--gates", "gates.yaml"],
            ("candidate.json", "gates.yaml"),
            "directory",
            "run_check",
        ),
        (
            ["check", "candidate.json", "--gates", "directory"],
            ("candidate.json",),
            "directory",
            "run_check",
        ),
    ],
)
def test_commands_reject_directory_read_input_before_calling_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    valid_files: tuple[str, ...],
    directory_name: str,
    boundary: str,
) -> None:
    """A directory supplied where a file is required must be a CLI usage error."""
    monkeypatch.chdir(tmp_path)
    _write_input_files(*valid_files)
    (tmp_path / directory_name).mkdir()
    monkeypatch.setattr(cli, boundary, _service_must_not_run)

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 2
    assert "is a directory" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "required_terms"),
    [
        (["ingest", "--help"], ("--config",)),
        (["evaluate", "--help"], ("DATASET", "--config", "--output")),
        (["compare", "--help"], ("BASELINE", "CANDIDATE")),
        (["sweep", "--help"], ("DATASET", "--config", "--size", "--output")),
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


def _write_input_files(*names: str) -> None:
    for name in names:
        Path(name).write_text("fixture", encoding="utf-8")


def _service_must_not_run(*args: object, **kwargs: object) -> None:
    raise AssertionError("CLI read-path validation should run before the service boundary")
