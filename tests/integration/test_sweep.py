from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_regress import cli, experiments
from rag_regress.comparison import compare_runs
from rag_regress.config import index_config_fingerprint, load_pipeline_config
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


def test_grid_reuses_indexes_and_publishes_distinct_comparisons(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)
    original = config.read_bytes()

    class CountingEmbedder(DeterministicTestEmbedder):
        document_calls = 0

        def embed_documents(self, texts):
            self.document_calls += 1
            return super().embed_documents(texts)

        def embed_query(self, text):
            return DeterministicTestEmbedder.embed_documents(self, [text])

    embedder = CountingEmbedder()
    loads = []

    def build(config):
        loads.append(config)
        return embedder

    monkeypatch.setattr(experiments, "build_embedder", build)
    output = tmp_path / "grid"
    runs = experiments.sweep(config, dataset, (12, 6), output, overlaps=(0, 1), top_ks=(2, 1))
    parameters = [
        (run.pipeline.chunking.size, run.pipeline.chunking.overlap, run.pipeline.retrieval.top_k)
        for run in runs
    ]
    assert parameters == [
        (12, 0, 2),
        (12, 0, 1),
        (12, 1, 2),
        (12, 1, 1),
        (6, 0, 2),
        (6, 0, 1),
        (6, 1, 2),
        (6, 1, 1),
    ]
    assert len(loads) == 1 and embedder.document_calls == 4
    assert config.read_bytes() == original
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["baseline"] == {"size": 12, "overlap": 0, "top_k": 2}
    assert len(summary["runs"]) == 8
    for index, (row, run, values) in enumerate(zip(summary["runs"], runs, parameters, strict=True)):
        size, overlap, top_k = values
        filename = f"size-{size}-overlap-{overlap}-top-k-{top_k}.json"
        assert row["artifact"] == filename
        assert (row["size"], row["overlap"], row["top_k"]) == values
        assert RunArtifact.model_validate_json((output / filename).read_text()) == run
        assert row["metrics"] == run.metrics.model_dump(mode="json")
        assert row["comparison"] == (
            compare_runs(runs[0], run).model_dump(mode="json") if index else None
        )
    assert len(list(output.glob("*.json"))) == 9
    base = load_pipeline_config(config)
    fingerprints = [
        index_config_fingerprint(
            base.model_copy(
                update={
                    "chunking": run.pipeline.chunking,
                    "retrieval": run.pipeline.retrieval,
                }
            )
        )
        for run in runs
    ]
    assert fingerprints[0] == fingerprints[1]
    assert len(set(fingerprints)) == 4
    experiments.sweep(config, dataset, (12, 6), tmp_path / "cached", overlaps=(0, 1), top_ks=(2, 1))
    assert embedder.document_calls == 4


@pytest.mark.parametrize(
    "axes",
    [
        {"overlaps": ()},
        {"overlaps": (-1,)},
        {"overlaps": (0, 0)},
        {"overlaps": (6,)},
        {"top_ks": ()},
        {"top_ks": (0,)},
        {"top_ks": (-1,)},
        {"top_ks": (2, 2)},
    ],
)
def test_invalid_grid_fails_before_model_or_writes(tmp_path, monkeypatch, axes):
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    with pytest.raises(UserInputError):
        experiments.sweep(config, dataset, (12, 6), tmp_path / "grid", **axes)
    assert not (tmp_path / "grid").exists()
    assert not (tmp_path / "state").exists()


def test_single_size_top_k_grid_uses_config_overlap_and_latency(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)
    config.write_text(config.read_text() + "latency: {warmup_queries: 3, repetitions: 2}\n")

    class CountingEmbedder(DeterministicTestEmbedder):
        queries = 0

        def embed_query(self, text):
            self.queries += 1
            return super().embed_query(text)

    embedder = CountingEmbedder()
    monkeypatch.setattr(experiments, "build_embedder", lambda config: embedder)
    runs = experiments.sweep(config, dataset, (12,), tmp_path / "grid", top_ks=(1, 2))
    assert len(runs) == 2 and embedder.queries == 14
    assert all(run.pipeline.chunking.overlap == 0 for run in runs)
    assert all(len(case.retrieval_samples_ms) == 2 for run in runs for case in run.cases)
    with pytest.raises(UserInputError, match="two distinct"):
        experiments.sweep(config, dataset, (12,), tmp_path / "single", top_ks=(2,))


def test_grid_late_failure_and_existing_symlink_preserve_outputs(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)

    class FailingEmbedder(DeterministicTestEmbedder):
        calls = 0

        def embed_query(self, text):
            self.calls += 1
            if self.calls == 5:
                raise RuntimeError("late grid failure")
            return super().embed_query(text)

    monkeypatch.setattr(experiments, "build_embedder", lambda config: FailingEmbedder())
    output = tmp_path / "grid"
    with pytest.raises(RuntimeError, match="late grid failure"):
        experiments.sweep(config, dataset, (12,), output, top_ks=(2, 1))
    assert not output.exists() and list(tmp_path.glob(".grid.*.tmp")) == []
    target = tmp_path / "previous"
    target.mkdir()
    marker = target / "summary.json"
    marker.write_text("previous result")
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    with pytest.raises(UserInputError, match="exists"):
        experiments.sweep(config, dataset, (12,), output, top_ks=(2, 1))
    assert marker.read_text() == "previous result"


def test_grid_artifacts_work_with_compare_report_and_gates(tmp_path, monkeypatch):
    from rag_regress.quality_gates import QualityGateConfig, check_quality_gates
    from rag_regress.reporting import render_comparison_markdown

    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", lambda config: DeterministicTestEmbedder())
    runs = experiments.sweep(config, dataset, (12,), tmp_path / "grid", top_ks=(2, 1))
    gates = QualityGateConfig.model_validate({"schema_version": 1, "minimum": {"mrr": 0.0}})
    for run in runs:
        assert compare_runs(runs[0], run).metrics.mrr.candidate == run.metrics.mrr
        gate_result = check_quality_gates(run, gates, runs[0])
        assert gate_result.passed
        rendered = render_comparison_markdown(runs[0], run, gates=gate_result)
        assert "mrr" in rendered.lower()


def test_grid_publication_failure_cleans_temporary_directory(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", lambda config: DeterministicTestEmbedder())
    original_write = experiments.atomic_write_json

    def fail_summary(path, payload):
        if path.name == "summary.json":
            raise OSError("summary write failed")
        original_write(path, payload)

    monkeypatch.setattr(experiments, "atomic_write_json", fail_summary)
    with pytest.raises(OSError, match="summary write failed"):
        experiments.sweep(config, dataset, (12,), tmp_path / "grid", top_ks=(2, 1))
    assert not (tmp_path / "grid").exists()
    assert list(tmp_path.glob(".grid.*.tmp")) == []


@pytest.mark.parametrize("axes", [{"overlaps": (False,)}, {"top_ks": (1.5,)}, {"top_ks": ("2",)}])
def test_grid_rejects_non_integer_axis_values(tmp_path, monkeypatch, axes):
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    with pytest.raises(UserInputError, match="integer"):
        experiments.sweep(config, dataset, (12, 6), tmp_path / "grid", **axes)
    assert not (tmp_path / "state").exists()


def test_grid_cli_reports_all_axes_and_variant_count(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", lambda config: DeterministicTestEmbedder())
    output = tmp_path / "grid"
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
            "--overlap",
            "0",
            "--overlap",
            "1",
            "--top-k",
            "2",
            "--top-k",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("Planned variants: 8\n")
    assert "size=12 overlap=0 top_k=2:" in result.stdout
    assert "size=6 overlap=1 top_k=1:" in result.stdout
    assert len(json.loads((output / "summary.json").read_text())["runs"]) == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--overlap", "6"],
        ["--top-k", "0"],
        ["--overlap", "0", "--overlap", "0"],
        ["--top-k", "2", "--top-k", "2"],
    ],
)
def test_grid_cli_invalid_axes_exit_two_before_model(tmp_path, monkeypatch, extra):
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
            "6",
            "--output",
            str(tmp_path / "grid"),
            *extra,
        ],
    )
    assert result.exit_code == 2
    assert "No such option" not in result.stderr
    assert not (tmp_path / "grid").exists()


@pytest.mark.parametrize("sizes", [(12, 0), (12, -1)])
def test_legacy_sweep_preserves_nonpositive_size_error(tmp_path, monkeypatch, sizes):
    config, dataset = _write_fixture_inputs(tmp_path)
    monkeypatch.setattr(experiments, "build_embedder", _must_not_load)
    with pytest.raises(UserInputError) as error:
        experiments.sweep(config, dataset, sizes, tmp_path / "sweep")
    assert str(error.value) == (
        "Each chunk size must be positive and greater than chunking.overlap"
    )
