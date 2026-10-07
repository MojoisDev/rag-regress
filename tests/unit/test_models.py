from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from rag_regress.models import (
    AggregateMetrics,
    CaseMetrics,
    CaseResult,
    RetrievalResult,
)


@pytest.mark.parametrize("score", [nan, inf, -inf])
def test_retrieval_result_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError, match="score"):
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="document-1",
            document_path="guide.md",
            rank=1,
            score=score,
            text="Guide text",
        )


def test_case_result_rejects_non_finite_latency() -> None:
    with pytest.raises(ValidationError, match="retrieval_ms"):
        CaseResult(
            id="case",
            question="Question?",
            expected_documents=("guide.md",),
            expected_text=(),
            results=(),
            metrics=CaseMetrics(
                recall_at_k=1.0,
                hit_at_k=1.0,
                reciprocal_rank=1.0,
                evidence_hit=None,
            ),
            retrieval_ms=inf,
        )


def test_aggregate_metrics_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="p95_retrieval_ms"):
        AggregateMetrics(
            recall_at_k=1.0,
            hit_rate=1.0,
            mrr=1.0,
            evidence_hit_rate=None,
            p50_retrieval_ms=1.0,
            p95_retrieval_ms=inf,
        )
