from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_regress.errors import UserInputError
from rag_regress.quality_gates import (
    QualityGateConfig,
    check_quality_gates,
    load_quality_gates,
)
from tests.unit.test_comparison import make_run


def test_absolute_and_relative_gates_report_every_failure() -> None:
    gates = QualityGateConfig.model_validate(
        {
            "schema_version": 1,
            "minimum": {"recall_at_k": 0.9, "mrr": 0.8},
            "maximum": {"p95_retrieval_ms": 100, "regressed_cases": 0},
            "allowed_drop": {"recall_at_k": 0.01, "mrr": 0.01},
        }
    )
    baseline = make_run(recall=0.95, mrr=0.9, p95=80)
    candidate = make_run(recall=0.80, mrr=0.70, p95=140)

    report = check_quality_gates(candidate, gates, baseline)

    assert report.passed is False
    assert {failure.gate for failure in report.failures} == {
        "minimum.recall_at_k",
        "minimum.mrr",
        "maximum.p95_retrieval_ms",
        "allowed_drop.recall_at_k",
        "allowed_drop.mrr",
    }


def test_relative_gate_without_baseline_is_invalid() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": 0.01}}
    )

    with pytest.raises(UserInputError, match="baseline"):
        check_quality_gates(make_run(), gates, baseline=None)


def test_gate_boundaries_are_inclusive() -> None:
    gates = QualityGateConfig.model_validate(
        {
            "schema_version": 1,
            "minimum": {"recall_at_k": 0.8},
            "maximum": {"p95_retrieval_ms": 100, "regressed_cases": 1},
            "allowed_drop": {"mrr": 0.1},
        }
    )
    baseline = make_run(
        mrr=0.8,
        case_reciprocal_ranks={"case": 1.0, "other": 1.0},
        case_ids=("case", "other"),
    )
    candidate = make_run(
        recall=0.8,
        mrr=0.7,
        p95=100,
        case_reciprocal_ranks={"case": 0.5, "other": 1.0},
        case_ids=("case", "other"),
    )

    assert check_quality_gates(candidate, gates, baseline).passed is True


def test_allowed_drop_allows_equality_roundoff_at_the_boundary() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": 0.1}}
    )

    report = check_quality_gates(make_run(mrr=0.7), gates, make_run(mrr=0.8))

    assert report.passed is True


def test_allowed_drop_rejects_next_float_below_an_exact_decimal_boundary() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": 0.25}}
    )
    baseline = make_run(mrr=0.75)

    assert check_quality_gates(make_run(mrr=0.5), gates, baseline).passed is True

    report = check_quality_gates(make_run(mrr=math.nextafter(0.5, 0.0)), gates, baseline)

    assert report.passed is False
    assert report.failures[0].gate == "allowed_drop.mrr"


def test_allowed_drop_rejects_a_genuine_excess_beyond_roundoff() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": 0.1}}
    )
    candidate_mrr = (0.8 - 0.1) - 5e-13

    report = check_quality_gates(make_run(mrr=candidate_mrr), gates, make_run(mrr=0.8))

    assert report.passed is False
    assert report.failures[0].gate == "allowed_drop.mrr"


def test_allowed_drop_failure_reports_full_precision_boundary() -> None:
    baseline_mrr = 0.8
    allowed_drop = 0.1333333333333333
    boundary = baseline_mrr - allowed_drop
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": allowed_drop}}
    )

    report = check_quality_gates(make_run(mrr=0.6), gates, make_run(mrr=baseline_mrr))

    assert report.passed is False
    assert report.failures[0].expected == f">= {repr(boundary)}"


def test_config_rejects_unknown_fields_invalid_values_and_empty_gate_files() -> None:
    invalid_configs = (
        {"schema_version": 1},
        {"schema_version": 1, "minimum": {"mrr": 1.1}},
        {"schema_version": 1, "maximum": {"p50_retrieval_ms": -1}},
        {"schema_version": 1, "maximum": {"regressed_cases": -1}},
        {"schema_version": 1, "unknown": {}},
        {"schema_version": 1, "minimum": {"unknown": 0.1}},
    )

    for value in invalid_configs:
        with pytest.raises(ValidationError):
            QualityGateConfig.model_validate(value)


def test_unmeasured_evidence_metric_fails_an_explicit_quality_gate() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "minimum": {"evidence_hit_rate": 0.8}}
    )

    report = check_quality_gates(make_run(evidence=None), gates)

    assert report.passed is False
    assert report.failures[0].gate == "minimum.evidence_hit_rate"
    assert report.failures[0].actual is None


def test_load_quality_gates_wraps_file_and_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "quality-gates.yaml"
    path.write_text("schema_version: 1\nminimum: []\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="Invalid quality gate config"):
        load_quality_gates(path)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_load_quality_gates_rejects_non_finite_limits(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / "quality-gates.yaml"
    path.write_text(
        f"schema_version: 1\nmaximum:\n  p95_retrieval_ms: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="p95_retrieval_ms"):
        load_quality_gates(path)
