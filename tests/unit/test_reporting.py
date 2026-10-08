from __future__ import annotations

import pytest

from rag_regress import reporting
from rag_regress.config import LatencyConfig
from rag_regress.models import GateFailure, GateReport, RetrievalResult
from tests.unit.test_comparison import make_run


def test_report_shows_metric_deltas_and_runtime_without_optional_evidence() -> None:
    baseline = make_run(mrr=1.0, p95=20)
    candidate = make_run(mrr=0.5, p95=25)
    text = reporting.render_comparison_markdown(baseline, candidate)

    assert "| mrr | 1.0 | 0.5 | -0.5 |" in text
    assert "| p95_retrieval_ms | 20.0 | 25.0 | 5.0 |" in text
    assert "| evidence_hit_rate | n/a | n/a | n/a |" in text
    assert "python_version" in text
    assert "embedding_device" in text
    assert "Quality gates: not supplied" in text


@pytest.mark.parametrize(
    ("metric", "candidate_value", "expected_row"),
    [
        ("recall_at_k", 0.5, "| recall_at_k | 1.0 | 0.5 | -0.5 |"),
        ("evidence_hit", 0.0, "| evidence_hit | 1.0 | 0.0 | -1.0 |"),
    ],
)
def test_report_includes_quality_decrease_even_when_reciprocal_rank_unchanged(
    metric: str,
    candidate_value: float,
    expected_row: str,
) -> None:
    baseline = make_run()
    baseline_case = baseline.cases[0].model_copy(
        update={"metrics": baseline.cases[0].metrics.model_copy(update={"evidence_hit": 1.0})}
    )
    baseline = baseline.model_copy(update={"cases": (baseline_case,)})
    candidate_case = baseline.cases[0].model_copy(
        update={"metrics": baseline.cases[0].metrics.model_copy(update={metric: candidate_value})}
    )
    candidate = baseline.model_copy(update={"cases": (candidate_case,)})
    text = reporting.render_comparison_markdown(baseline, candidate)

    assert "Questions with quality decreases: 1" in text
    assert expected_row in text
    assert "Question for case" in text
    assert "No retrieved passages" in text


def test_report_renders_question_and_passages_as_literal_markdown_blocks() -> None:
    baseline = make_run()
    passage = RetrievalResult(
        chunk_id="chunk",
        document_id="doc",
        document_path="[link](https://bad)",
        rank=1,
        score=0.9,
        text="```\n<script>bad</script>\n# injected\n````",
    )
    case = baseline.cases[0].model_copy(
        update={
            "question": "</details>\n# hostile",
            "results": (passage,),
            "metrics": baseline.cases[0].metrics.model_copy(update={"reciprocal_rank": 0.0}),
        }
    )
    candidate = baseline.model_copy(update={"cases": (case,)})
    text = reporting.render_comparison_markdown(baseline, candidate)

    assert "    </details>\n    # hostile" in text
    assert "    ```\n    <script>bad</script>\n    # injected\n    ````" in text
    assert "    [link](https://bad)" in text
    assert "\n<script>" not in text
    assert "\n# injected" not in text
    assert "Rank: 1" in text
    assert "Score: 0.9" in text


@pytest.mark.parametrize("passed", [True, False])
def test_report_preserves_gate_outcome_and_every_failure(passed: bool) -> None:
    failures = (
        ()
        if passed
        else (
            GateFailure(gate="minimum.mrr", expected=">= 0.8", actual=0.5),
            GateFailure(gate="allowed_drop.mrr", expected=">= 0.9", actual=0.5, baseline=1.0),
        )
    )
    text = reporting.render_comparison_markdown(
        make_run(), make_run(mrr=0.5), GateReport(passed=passed, failures=failures)
    )
    assert ("Quality gates: PASS" if passed else "Quality gates: FAIL") in text
    for failure in failures:
        assert failure.gate in text
        assert failure.expected in text


def test_report_with_no_quality_decreases_has_no_passage_details() -> None:
    run = make_run()
    text = reporting.render_comparison_markdown(run, run)
    assert "Questions with quality decreases: 0" in text
    assert "No questions have decreased quality metrics" in text


def test_report_rejects_incompatible_runs() -> None:
    from rag_regress.errors import UserInputError

    with pytest.raises(UserInputError, match="dataset_fingerprint"):
        reporting.render_comparison_markdown(make_run(), make_run(dataset_fingerprint="other"))


def test_report_labels_repeated_latency_and_differing_measurement_settings() -> None:
    baseline = make_run()
    baseline = baseline.model_copy(
        update={
            "pipeline": baseline.pipeline.model_copy(
                update={
                    "latency": LatencyConfig(warmup_queries=1, repetitions=2),
                }
            ),
        }
    )
    candidate = baseline.model_copy(
        update={
            "pipeline": baseline.pipeline.model_copy(
                update={
                    "latency": LatencyConfig(warmup_queries=2, repetitions=3),
                }
            ),
            "cases": (
                baseline.cases[0].model_copy(
                    update={
                        "retrieval_samples_ms": (1.0, 2.0, 3.0),
                        "metrics": baseline.cases[0].metrics.model_copy(
                            update={"reciprocal_rank": 0.5}
                        ),
                    }
                ),
            ),
        }
    )
    text = reporting.render_comparison_markdown(baseline, candidate)
    assert "Latency measurement settings differ" in text
    assert '"repetitions": 2' in text
    assert '"repetitions": 3' in text
    assert "p50 retrieval ms" in text
    assert "Legacy run" not in text


def test_report_quality_decreases_are_sorted_by_case_id() -> None:
    baseline = make_run(case_ids=("z", "a"))
    candidate = make_run(case_ids=("z", "a"), case_reciprocal_ranks={"z": 0.5, "a": 0.0})
    text = reporting.render_comparison_markdown(baseline, candidate)
    assert text.index("Question for a") < text.index("Question for z")
