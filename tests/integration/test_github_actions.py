from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from typer.testing import CliRunner

from rag_regress.cli import app
from tests.unit.test_comparison import make_run


@pytest.fixture
def inputs(tmp_path: Path) -> list[str]:
    baseline = make_run()
    candidate = make_run(mrr=0.5, case_reciprocal_ranks={"case": 0.5})
    (tmp_path / "baseline.json").write_text(baseline.model_dump_json())
    (tmp_path / "candidate.json").write_text(candidate.model_dump_json())
    (tmp_path / "gates.yaml").write_text("schema_version: 1\nminimum:\n  mrr: 0.8\n")
    return [
        "check",
        str(tmp_path / "candidate.json"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--gates",
        str(tmp_path / "gates.yaml"),
    ]


def test_actions_reports_regression_and_failed_gate_and_appends_summary(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("Previous step content\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 1, result.output
    assert "::warning::Case case:" in result.stdout
    assert "reciprocal_rank baseline=1.0 candidate=0.5" in result.stdout
    assert "::error::minimum.mrr: expected >= 0.8, actual 0.5" in result.stdout
    text = summary.read_text()
    assert text.startswith("Previous step content\n")
    assert "| mrr | 1.0 | 0.5 | -0.5 |" in text
    assert "Quality gates: FAIL" in text
    assert "minimum.mrr" in text
    assert "case" in text
    assert "Question for case" not in text


@pytest.mark.parametrize("with_baseline", [True, False])
def test_actions_passes_without_summary_environment(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_baseline: bool,
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    (tmp_path / "gates.yaml").write_text("schema_version: 1\nminimum:\n  mrr: 0\n")
    if not with_baseline:
        del inputs[2:4]
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 0, result.output
    assert "PASS: all quality gates passed" in result.stdout
    assert "::error::" not in result.stdout
    assert ("::warning::" in result.stdout) == with_baseline


def test_normal_check_does_not_write_actions_summary(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, inputs)
    assert result.exit_code == 1
    assert "::error::" not in result.stdout
    assert not summary.exists()


@pytest.mark.parametrize("target", ["directory", "candidate.json", "hardlink", "symlink"])
def test_actions_rejects_invalid_summary_without_damaging_inputs(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    candidate = tmp_path / "candidate.json"
    original = candidate.read_bytes()
    summary = tmp_path / target
    if target == "directory":
        summary.mkdir()
    elif target == "hardlink":
        summary.hardlink_to(candidate)
    elif target == "symlink":
        summary.symlink_to(candidate)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 2, result.output
    assert "summary" in result.stderr.lower()
    assert candidate.read_bytes() == original
    assert "::warning::" not in result.stdout


def test_actions_escapes_case_ids_and_warns_for_recall_decrease(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "hostile%\r\n::error::injected <script>|"
    baseline = make_run(case_ids=(case_id,))
    case = baseline.cases[0]
    candidate = baseline.model_copy(
        update={
            "cases": (
                case.model_copy(
                    update={
                        "metrics": case.metrics.model_copy(update={"recall_at_k": 0.0}),
                    }
                ),
            )
        }
    )
    (tmp_path / "baseline.json").write_text(baseline.model_dump_json())
    (tmp_path / "candidate.json").write_text(candidate.model_dump_json())
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 0, result.output
    assert "hostile%25%0D%0A::error::injected" in result.stdout
    assert not any(line.startswith("::error::") for line in result.stdout.splitlines())
    assert "recall_at_k baseline=1.0 candidate=0.0" in result.stdout
    assert "<script>" not in MarkdownIt().render(summary.read_text())


def test_actions_rejects_incompatible_runs_before_summary(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "candidate.json").write_text(
        make_run(dataset_fingerprint="other").model_dump_json()
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 2
    assert "dataset_fingerprint" in result.stderr
    assert not summary.exists()


def test_actions_bounds_large_regression_output(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = tuple(f"{i:03}-" + "<" * 5000 for i in range(110))
    baseline = make_run(case_ids=ids)
    candidate = make_run(case_ids=ids, case_reciprocal_ranks=dict.fromkeys(ids, 0.5))
    (tmp_path / "baseline.json").write_text(baseline.model_dump_json())
    (tmp_path / "candidate.json").write_text(candidate.model_dump_json())
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("::warning::") == 10
    assert "100 additional cases" in result.stdout
    text = summary.read_text()
    assert "Quality decreases: 110" in text
    assert "10 additional cases" in text
    assert summary.stat().st_size < 1024 * 1024


def test_actions_summary_without_baseline_has_candidate_metrics(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del inputs[2:4]
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 1
    assert "Baseline: not supplied" in summary.read_text()
    assert "| mrr | n/a | 0.5 | n/a |" in summary.read_text()


def test_actions_rejects_oversized_summary_before_append(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.md"
    original = "x" * (1024 * 1024)
    summary.write_text(original)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 2
    assert "1 MiB" in result.stderr
    assert summary.read_text() == original
    assert "::warning::" not in result.stdout


def test_actions_escapes_multiline_invalid_input_errors(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "broken%\n::error::injected.json"
    path.write_text("not json")
    inputs[1] = str(path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 2
    assert "broken%25%0A::error::injected.json" in result.stderr
    assert result.stderr.startswith("::error::Error: Invalid run artifact")
    assert len(result.stderr.splitlines()) == 1
    assert not summary.exists()


def test_actions_unchanged_quality_has_no_warning(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "candidate.json").write_text(make_run().model_dump_json())
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 0
    assert "::warning::" not in result.stdout
    assert "::error::" not in result.stdout


def test_actions_caps_error_annotations_and_retains_all_failed_gates(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "baseline.json").write_text(make_run(evidence=1).model_dump_json())
    (tmp_path / "candidate.json").write_text(
        make_run(
            recall=0.5,
            hit_rate=0.5,
            mrr=0.5,
            evidence=0,
            case_reciprocal_ranks={"case": 0.5},
        ).model_dump_json()
    )
    (tmp_path / "gates.yaml").write_text(
        "schema_version: 1\n"
        "minimum: {recall_at_k: 1, hit_rate: 1, mrr: 1, evidence_hit_rate: 1}\n"
        "maximum: {p50_retrieval_ms: 0, p95_retrieval_ms: 0, regressed_cases: 0}\n"
        "allowed_drop: {recall_at_k: 0, hit_rate: 0, mrr: 0, evidence_hit_rate: 0}\n"
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 1, result.output
    assert result.stdout.count("::error::") == 10
    assert "1 additional gates failed" in result.stdout
    text = summary.read_text()
    assert "maximum.regressed_cases" in text
    assert "allowed_drop.evidence_hit_rate" in text
    assert "baseline 1.0" in text


@pytest.mark.parametrize("source", ["baseline.json", "gates.yaml"])
def test_actions_rejects_summary_aliases_of_other_inputs(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    path = tmp_path / source
    original = path.read_bytes()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 2
    assert "summary must not modify" in result.stderr
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "case_id",
    [
        "![forged result](https://example.com/image.png)",
        "[forged result](https://example.com)",
        "` ![forged result](https://example.com/image.png) `` <script>bad</script>",
    ],
)
def test_actions_renders_hostile_markdown_case_ids_literally(
    inputs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
) -> None:
    (tmp_path / "baseline.json").write_text(make_run(case_ids=(case_id,)).model_dump_json())
    (tmp_path / "candidate.json").write_text(
        make_run(
            case_ids=(case_id,),
            case_reciprocal_ranks={case_id: 0.5},
        ).model_dump_json()
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(app, [*inputs, "--github-actions"])
    assert result.exit_code == 0, result.output
    rendered = MarkdownIt().render(summary.read_text())
    assert "<img " not in rendered
    assert "<a " not in rendered
    assert "<script>" not in rendered
