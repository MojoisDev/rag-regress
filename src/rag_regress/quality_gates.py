"""Strict, local quality-gate loading and evaluation."""

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_regress.comparison import compare_runs
from rag_regress.errors import UserInputError
from rag_regress.models import ComparisonReport, GateFailure, GateReport, RunArtifact

_ALLOWED_DROP_ABSOLUTE_TOLERANCE = 1e-12


class QualityMetricLimits(BaseModel):
    """Optional inclusive limits for aggregate quality metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class MaximumGateLimits(BaseModel):
    """Optional inclusive upper limits for latency and regression count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p50_retrieval_ms: float | None = Field(default=None, ge=0.0)
    p95_retrieval_ms: float | None = Field(default=None, ge=0.0)
    regressed_cases: int | None = Field(default=None, ge=0)


class QualityGateConfig(BaseModel):
    """Versioned, strict quality-gate configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    minimum: QualityMetricLimits = Field(default_factory=QualityMetricLimits)
    maximum: MaximumGateLimits = Field(default_factory=MaximumGateLimits)
    allowed_drop: QualityMetricLimits = Field(default_factory=QualityMetricLimits)

    @model_validator(mode="after")
    def reject_empty_gates(self) -> "QualityGateConfig":
        if not any(
            value is not None
            for section in (self.minimum, self.maximum, self.allowed_drop)
            for value in section.model_dump().values()
        ):
            raise ValueError("quality gate configuration must contain at least one gate")
        return self

    @property
    def requires_baseline(self) -> bool:
        """Return whether this config includes a relative gate."""
        return self.maximum.regressed_cases is not None or any(
            value is not None for value in self.allowed_drop.model_dump().values()
        )


def load_quality_gates(path: Path) -> QualityGateConfig:
    """Load one strict YAML quality-gate configuration."""
    source = path.resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return QualityGateConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise UserInputError(f"Invalid quality gate config {source}: {exc}") from exc


def check_quality_gates(
    candidate: RunArtifact,
    gates: QualityGateConfig,
    baseline: RunArtifact | None = None,
) -> GateReport:
    """Evaluate every applicable absolute and relative gate."""
    if gates.requires_baseline and baseline is None:
        raise UserInputError("A baseline run is required by relative quality gates")

    comparison = compare_runs(baseline, candidate) if baseline is not None else None
    failures = _evaluate_absolute_gates(candidate, gates)
    failures.extend(_evaluate_relative_gates(candidate, baseline, comparison, gates))
    return GateReport(passed=not failures, failures=tuple(failures))


def _evaluate_absolute_gates(candidate: RunArtifact, gates: QualityGateConfig) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for metric, minimum in gates.minimum.model_dump().items():
        if minimum is None:
            continue
        actual = getattr(candidate.metrics, metric)
        if actual is None or actual < minimum:
            failures.append(
                GateFailure(
                    gate=f"minimum.{metric}", expected=f">= {minimum}", actual=actual
                )
            )
    for metric, maximum in gates.maximum.model_dump().items():
        if metric == "regressed_cases" or maximum is None:
            continue
        actual = getattr(candidate.metrics, metric)
        if actual > maximum:
            failures.append(
                GateFailure(
                    gate=f"maximum.{metric}", expected=f"<= {maximum}", actual=actual
                )
            )
    return failures


def _evaluate_relative_gates(
    candidate: RunArtifact,
    baseline: RunArtifact | None,
    comparison: ComparisonReport | None,
    gates: QualityGateConfig,
) -> list[GateFailure]:
    if baseline is None:
        return []

    failures: list[GateFailure] = []
    for metric, allowed_drop in gates.allowed_drop.model_dump().items():
        if allowed_drop is None:
            continue
        candidate_value = getattr(candidate.metrics, metric)
        baseline_value = getattr(baseline.metrics, metric)
        if candidate_value is None or baseline_value is None or _drops_beyond_allowed(
            candidate_value, baseline_value, allowed_drop
        ):
            boundary = baseline_value - allowed_drop if baseline_value is not None else None
            expected = (
                f">= {boundary:g}"
                if boundary is not None
                else f"baseline - {allowed_drop}"
            )
            failures.append(
                GateFailure(
                    gate=f"allowed_drop.{metric}",
                    expected=expected,
                    actual=candidate_value,
                    baseline=baseline_value,
                )
            )

    if gates.maximum.regressed_cases is not None:
        if comparison is None:
            raise AssertionError("comparison is required for relative quality gates")
        regressed_cases = len(comparison.regressed_case_ids)
        if regressed_cases > gates.maximum.regressed_cases:
            failures.append(
                GateFailure(
                    gate="maximum.regressed_cases",
                    expected=f"<= {gates.maximum.regressed_cases}",
                    actual=float(regressed_cases),
                )
            )
    return failures


def _drops_beyond_allowed(candidate: float, baseline: float, allowed_drop: float) -> bool:
    """Return whether a quality drop exceeds its limit beyond float roundoff."""
    boundary = baseline - allowed_drop
    return candidate < boundary and not math.isclose(
        candidate,
        boundary,
        rel_tol=0.0,
        abs_tol=_ALLOWED_DROP_ABSOLUTE_TOLERANCE,
    )
