"""FlowState dataclass and invariant checking."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

TrackId = Literal["T1", "T2", "T3"]


class InvariantViolationError(Exception):
    """Raised when a FlowState violates one of the assert_invariants checks."""


class FlowState(BaseModel):
    """Joint state of activation, parameters, curriculum at consolidation step tau."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rho: dict[str, np.ndarray]
    P_theta: np.ndarray
    mu_curr: np.ndarray
    tau: int
    metadata: dict[str, Any]

    @field_validator("metadata")
    @classmethod
    def validate_track_id(cls, v: dict[str, Any]) -> dict[str, Any]:
        track_id = v.get("track_id")
        if track_id not in {"T1", "T2", "T3"}:
            raise ValueError(f"track_id must be T1/T2/T3, got {track_id!r}")
        return v

    @field_validator("tau")
    @classmethod
    def validate_tau_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"tau must be non-negative, got {v}")
        return v


def assert_invariants(state: FlowState, mass_tol: float = 1e-4, neg_tol: float = 1e-6) -> None:
    """Verify FlowState invariants. Raise InvariantViolationError on failure."""
    for name, rho in state.rho.items():
        if not np.isfinite(rho).all():
            raise InvariantViolationError(f"NaN/Inf in {name}")
        if (rho < -neg_tol).any():
            raise InvariantViolationError(f"Negative density in {name}: min={rho.min()}")
        mass = float(rho.sum())
        if abs(mass - 1.0) > mass_tol:
            raise InvariantViolationError(f"Mass not conserved in {name}: sum={mass}")
