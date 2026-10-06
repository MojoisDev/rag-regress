"""Canonical hashing helpers."""

import hashlib
import json


def hash_canonical(payload: object) -> str:
    """Return a stable SHA-256 hash for JSON-compatible data."""
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
