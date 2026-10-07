"""Shared strict validation conventions for configuration and artifact models."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Reject unknown fields and non-finite numbers in immutable public models."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
