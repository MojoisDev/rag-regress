from pathlib import Path

import pytest

from rag_regress import artifacts
from rag_regress.artifacts import atomic_write_json


def test_atomic_json_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "run.json"
    target.write_text('{"state":"old"}', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(target, {"state": "new"})

    assert target.read_text(encoding="utf-8") == '{"state":"old"}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_is_canonical_utf8_with_a_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "run.json"

    atomic_write_json(target, {"z": "café", "a": {"b": 1}})

    assert target.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": 1\n  },\n  "z": "café"\n}\n'
    )


def test_atomic_text_failure_preserves_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("old report", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        artifacts.atomic_write_text(target, "new report")
    assert target.read_text(encoding="utf-8") == "old report"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_text_creates_parent_and_preserves_utf8_content(tmp_path: Path) -> None:
    target = tmp_path / "new" / "report.md"
    artifacts.atomic_write_text(target, "# café\n")
    assert target.read_text(encoding="utf-8") == "# café\n"


def test_atomic_json_serializes_incrementally_without_a_second_complete_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_full_string(*args: object, **kwargs: object) -> None:
        raise AssertionError("JSON serialization must stay incremental")

    monkeypatch.setattr(artifacts.json, "dumps", reject_full_string)
    target = tmp_path / "run.json"
    atomic_write_json(target, {"payload": ["café", 1]})
    assert '"café"' in target.read_text(encoding="utf-8")


def test_atomic_json_serialization_failure_preserves_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(TypeError):
        atomic_write_json(target, {"unsupported": object()})
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".*.tmp")) == []
