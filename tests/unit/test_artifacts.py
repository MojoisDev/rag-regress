from pathlib import Path

import pytest

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
