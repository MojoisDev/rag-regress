from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_regress import cli, experiments
from rag_regress.comparison import compare_runs
from rag_regress.config import load_pipeline_config
from rag_regress.errors import UserInputError
from rag_regress.index import bundle_path
from rag_regress.models import RunArtifact
from tests.helpers import DeterministicTestEmbedder
from tests.integration.test_experiments import _write_fixture_inputs


def test_sweep_writes_comparable_runs_and_loads_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    original_config = config.read_bytes()
    output = tmp_path / "sweep"
    loads = []

    def build(config):
        loads.append(config)
        return DeterministicTestEmbedder()

    monkeypatch.setattr(experiments, "build_embedder", build)
    runs = experiments.sweep(config, dataset, (12, 6), output)

    assert len(loads) == 1
    assert [run.pipeline.chunking.size for run in runs] == [12, 6]
    assert all(run.pipeline.chunking.overlap == 0 for run in runs)
    assert config.read_bytes() == original_config
    for size, run in zip((12, 6), runs, strict=True):
        assert RunArtifact.model_validate_json((output / f"size-{size}.json").read_text()) == run
        pipeline = load_pipeline_config(config)
        pipeline = pipeline.model_copy(
            update={"chunking": pipeline.chunking.model_copy(update={"size": size})}
        )
        assert bundle_path(pipeline, run.corpus_fingerprint).is_dir()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["baseline_size"] == 12
    assert summary["runs"][1]["artifact"] == "size-6.json"
    assert summary["runs"][1]["metrics"] == runs[1].metrics.model_dump(mode="json")
    assert summary["runs"][1]["comparison"] == compare_runs(*runs).model_dump(mode="json")


def test_sweep_applies_latency_settings_and_warms_every_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(config.read_text() + "latency: {warmup_queries: 3, repetitions: 2}\n")

    class CountingEmbedder(DeterministicTestEmbedder):
        calls = 0

        def embed_query(self, question):
            self.calls += 1
            return super().embed_query(question)

    embedder = CountingEmbedder()
    monkeypatch.setattr(experiments, "build_embedder", lambda config: embedder)
    runs = experiments.sweep(config, dataset, (12, 6), tmp_path / "sweep")
    assert embedder.calls == 14  # Each variant: three warmups plus two samples per case.
    assert all(run.pipeline.latency.repetitions == 2 for run in runs)
    assert all(len(case.retrieval_samples_ms) == 2 for run in runs for case in run.cases)


@pytest.mark.parametrize("sizes", [(), (6,), (6, 6), (6, 0), (6, -1)])
def test_sweep_rejects_invalid_sizes_before_loading_model_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sizes: tuple[int, ...]
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    with pytest.raises(UserInputError, match="size"):
        experiments.sweep(config, dataset, sizes, tmp_path / "sweep")
    assert not (tmp_path / "sweep").exists()
    assert not (tmp_path / "state").exists()


def test_sweep_rejects_size_not_exceeding_overlap(tmp_path: Path) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(config.read_text().replace("overlap: 0", "overlap: 5"))
    with pytest.raises(UserInputError, match="overlap"):
        experiments.sweep(config, dataset, (12, 5), tmp_path / "sweep")


def test_sweep_preserves_existing_output(tmp_path: Path) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    output = tmp_path / "sweep"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("previous summary")
    with pytest.raises(UserInputError, match="exists"):
        experiments.sweep(config, dataset, (12, 6), output)
    assert marker.read_text() == "previous summary"


def test_failed_sweep_does_not_publish_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingEmbedder(DeterministicTestEmbedder):
        calls = 0

        def embed_query(self, text):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("simulated query failure")
            return super().embed_query(text)

    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", lambda config: FailingEmbedder())
    output = tmp_path / "sweep"
    with pytest.raises(RuntimeError, match="simulated query failure"):
        experiments.sweep(config, dataset, (12, 6), output)
    assert not output.exists()
    assert list(tmp_path.glob(".sweep.*.tmp")) == []


@pytest.mark.parametrize("input_name", ["config", "dataset"])
def test_sweep_cli_rejects_missing_input_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_name: str
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    (config if input_name == "config" else dataset).unlink()
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    result = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            str(dataset),
            "--config",
            str(config),
            "--size",
            "12",
            "--size",
            "6",
            "--output",
            str(tmp_path / "sweep"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_sweep_cli_rejects_duplicate_sizes_with_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    result = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            str(dataset),
            "--config",
            str(config),
            "--size",
            "12",
            "--size",
            "12",
            "--output",
            str(tmp_path / "sweep"),
        ],
    )
    assert result.exit_code == 2
    assert "distinct chunk sizes" in result.stderr


def test_sweep_cli_reports_quality_latency_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", lambda config: DeterministicTestEmbedder())
    output = tmp_path / "sweep"
    result = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            str(dataset),
            "--config",
            str(config),
            "--size",
            "12",
            "--size",
            "6",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "size=12" in result.stdout and "size=6" in result.stdout
    assert "recall_at_k=" in result.stdout and "p95_retrieval_ms=" in result.stdout
    assert str(output / "summary.json") in result.stdout


def _must_not_load(*args, **kwargs):
    raise AssertionError("validation must finish before loading the model")
