import numpy as np
import pytest
from pydantic import ValidationError

from kiki_flow_core.state import FlowState, InvariantViolationError, assert_invariants


def test_flow_state_minimal_construction():
    state = FlowState(
        rho={"phono": np.array([0.5, 0.5])},
        P_theta=np.zeros(10),
        mu_curr=np.array([0.25, 0.25, 0.25, 0.25]),
        tau=0,
        metadata={"track_id": "T1", "step_id": "abc"},
    )
    assert state.tau == 0
    assert "phono" in state.rho


def test_flow_state_rejects_invalid_track_id():
    with pytest.raises(ValidationError):
        FlowState(
            rho={"phono": np.array([1.0])},
            P_theta=np.zeros(10),
            mu_curr=np.array([1.0]),
            tau=0,
            metadata={"track_id": "INVALID"},
        )


def test_assert_invariants_passes_on_valid_state():
    state = FlowState(
        rho={"phono": np.array([0.5, 0.5])},
        P_theta=np.zeros(10),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )
    assert_invariants(state)


def test_assert_invariants_detects_nan():
    state = FlowState(
        rho={"phono": np.array([float("nan"), 0.5])},
        P_theta=np.zeros(10),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )
    with pytest.raises(InvariantViolationError, match="NaN/Inf"):
        assert_invariants(state)


def test_assert_invariants_detects_negative_density():
    state = FlowState(
        rho={"phono": np.array([-0.1, 1.1])},
        P_theta=np.zeros(10),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )
    with pytest.raises(InvariantViolationError, match="Negative density"):
        assert_invariants(state)


def test_assert_invariants_detects_mass_violation():
    state = FlowState(
        rho={"phono": np.array([0.3, 0.3])},  # sum = 0.6
        P_theta=np.zeros(10),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )
    with pytest.raises(InvariantViolationError, match="Mass not conserved"):
        assert_invariants(state)
