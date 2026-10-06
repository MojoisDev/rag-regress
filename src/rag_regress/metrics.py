"""Pure retrieval metric calculations used by evaluation runs."""

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    """Return the fraction of expected documents represented in the first ``k`` chunks."""
    retrieved = set(ranked[:k])
    return len(retrieved & expected) / len(expected)


def hit_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    """Return one when any expected document occurs in the first ``k`` chunks."""
    return float(bool(set(ranked[:k]) & expected))


def reciprocal_rank(ranked: Sequence[str], expected: set[str]) -> float:
    """Return the reciprocal rank of the first relevant retrieved chunk."""
    for rank, document_path in enumerate(ranked, start=1):
        if document_path in expected:
            return 1.0 / rank
    return 0.0


def evidence_hit(chunks: Sequence[str], expected_text: Sequence[str]) -> float:
    """Return one only when every normalized evidence target appears in retrieved text."""
    combined = _normalize_match_text(" ".join(chunks))
    return float(all(_normalize_match_text(item) in combined for item in expected_text))


def nearest_rank_percentile(samples: Sequence[float], percentile: float) -> float:
    """Select a nearest-rank percentile from an unordered non-empty sample sequence."""
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must satisfy 0 < percentile <= 1")
    ordered = sorted(samples)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _normalize_match_text(value: str) -> str:
    """Normalize text for case-insensitive, whitespace-insensitive evidence matching."""
    return " ".join(value.split()).casefold()
