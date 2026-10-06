import pytest

from rag_regress.metrics import (
    evidence_hit,
    hit_at_k,
    nearest_rank_percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_document_metrics_deduplicate_multiple_chunks() -> None:
    paths = ["wrong.md", "right.md", "right.md", "other.md"]
    expected = {"right.md", "second.md"}

    assert recall_at_k(paths, expected, 4) == 0.5
    assert hit_at_k(paths, expected, 4) == 1.0
    assert reciprocal_rank(paths, expected) == 0.5


def test_document_metrics_respect_top_k_and_first_relevant_chunk_rank() -> None:
    paths = ["wrong.md", "first.md", "second.md"]
    expected = {"first.md", "second.md"}

    assert recall_at_k(paths, expected, 2) == 0.5
    assert hit_at_k(paths, expected, 1) == 0.0
    assert reciprocal_rank(paths, expected) == 0.5


def test_evidence_hit_requires_every_expected_substring() -> None:
    chunks = ["Reset links expire", "after   30 minutes."]

    assert evidence_hit(chunks, ["reset links", "after 30 minutes"]) == 1.0
    assert evidence_hit(chunks, ["reset links", "administrator approval"]) == 0.0


def test_evidence_hit_normalizes_case_and_whitespace_across_chunk_boundaries() -> None:
    assert evidence_hit(["The RESET", " link expires"], [" reset link "]) == 1.0


def test_nearest_rank_percentile() -> None:
    assert nearest_rank_percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0


def test_nearest_rank_percentile_sorts_samples_before_selecting_rank() -> None:
    assert nearest_rank_percentile([40.0, 10.0, 30.0, 20.0], 0.5) == 20.0


@pytest.mark.parametrize("percentile", [0.0, -0.1, 1.1])
def test_nearest_rank_percentile_rejects_invalid_percentiles(percentile: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        nearest_rank_percentile([1.0], percentile)
