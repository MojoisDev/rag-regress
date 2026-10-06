"""Canonical, atomic writers for human-readable artifact files."""

import json
import os
import uuid
from pathlib import Path


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Replace a JSON artifact only after its complete contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
