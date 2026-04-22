"""Property-based tests on the core math invariants.

Uses Hypothesis to drive the functions with many random inputs and
assert invariants that should hold by construction. These tests complement
the deterministic unit tests by catching bugs that manifest only on
specific input shapes or values.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from kiki_flow_core.master_equation import JKOStep, ZeroF
from kiki_flow_core.modules.advection_diffusion import AdvectionDiffusion
from kiki_flow_core.modules.phonological_loop import PhonologicalLoop
from kiki_flow_core.modules.scaffolding_scheduler import ScaffoldingScheduler
from kiki_flow_core.species import CanonicalSpecies, MixedCanonicalSpecies
from kiki_flow_core.state import FlowState, InvariantViolationError, assert_invariants

# Tolerances (named per tests/CLAUDE.md discipline)
MASS_TOL = 1e-4
NEG_TOL = 1e-6
BOUND_TOL = 1e-9

# Strategy: a simplex-valid probability vector of variable size
simplex_vector = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=32),
    elements=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
).map(lambda a: a / a.sum())


def _make_state(rho: np.ndarray) -> FlowState:
    return FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.full(rho.size, 1.0 / rho.size),
        tau=0,
        metadata={"track_id": "T2"},
    )


# ============================================================================
# State invariants
# ============================================================================


@given(rho=simplex_vector)
def test_assert_invariants_accepts_any_simplex_vector(rho: np.ndarray) -> None:
    """assert_invariants must accept every valid simplex vector."""
    assert_invariants(_make_state(rho))


@given(rho=simplex_vector)
def test_assert_invariants_rejects_nan_injection(rho: np.ndarray) -> None:
    """Any NaN in rho must trigger InvariantViolationError."""
    if rho.size < 1:
        return
    poisoned = rho.copy()
    poisoned[0] = float("nan")
    with pytest.raises(InvariantViolationError, match="NaN/Inf"):
        assert_invariants(_make_state(poisoned))


@given(
    rho=simplex_vector,
    scale=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
)
def test_assert_invariants_rejects_mass_violation(rho: np.ndarray, scale: float) -> None:
    """Any mass != 1 beyond tol must trigger InvariantViolationError."""
    poisoned = rho * scale
    with pytest.raises(InvariantViolationError, match="Mass not conserved"):
        assert_invariants(_make_state(poisoned))


# ============================================================================
# Module invariants
# ============================================================================


@given(
    h_min=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False),
    h_max=st.floats(min_value=1e-1, max_value=10.0, allow_nan=False),
    errors=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=16),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
)
def test_scheduler_h_always_within_bounds(h_min: float, h_max: float, errors: np.ndarray) -> None:
    """next_step must always return h in [h_min, h_max]."""
    s = ScaffoldingScheduler(h_min=h_min, h_max=h_max)
    h, mu = s.next_step(error_profile=errors)
    assert h_min - BOUND_TOL <= h <= h_max + BOUND_TOL
    assert abs(mu.sum() - 1.0) < MASS_TOL
    assert (mu >= -NEG_TOL).all()


@given(
    output=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=16),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    ),
    strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_phono_loop_source_always_bounded_by_strength(output: np.ndarray, strength: float) -> None:
    """PhonologicalLoop source magnitude must never exceed correction_strength."""
    loop = PhonologicalLoop(
        detector=lambda out: out,
        correction_strength=strength,  # noqa: PLW0108
    )
    rho = np.full(output.size, 1.0 / output.size)
    source = loop.source_term(rho_phono=rho, output=output)
    assert np.abs(source).max() <= strength + BOUND_TOL


# ============================================================================
# Advection-diffusion invariants
# ============================================================================


@given(
    rho=simplex_vector,
    dt=st.floats(min_value=1e-4, max_value=1e-2, allow_nan=False),
    diffusion=st.floats(min_value=0.0, max_value=0.05, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_advection_diffusion_preserves_mass_and_sign(
    rho: np.ndarray, dt: float, diffusion: float
) -> None:
    """One advection-diffusion step must preserve mass and non-negativity."""
    n = rho.size
    x = np.linspace(-1, 1, n)
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=diffusion)
    v = np.full(n, 0.05)
    out = solver.step_1d(rho, v_field=v, dt=dt)
    assert abs(out.sum() - 1.0) < MASS_TOL
    assert (out >= -NEG_TOL).all()
    assert np.isfinite(out).all()


# ============================================================================
# Species invariants
# ============================================================================


def test_ortho_species_coupling_shape_always_4x4() -> None:
    """CanonicalSpecies.coupling_matrix() must always be (4, 4)."""
    s = CanonicalSpecies()
    j = s.coupling_matrix()
    assert j.shape == (4, 4)
    assert np.isfinite(j).all()
    assert (j >= 0).all()


@given(
    n_stacks=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_hybrid_species_coupling_tensor_shape_invariant(n_stacks: int, seed: int) -> None:
    """MixedCanonicalSpecies.coupling_tensor() must have shape (4, n_stacks, 4, n_stacks)."""
    stacks = [f"stack_{i}" for i in range(n_stacks)]
    s = MixedCanonicalSpecies(stack_names=stacks, projection_init="random", seed=seed)
    t = s.coupling_tensor()
    assert t.shape == (4, n_stacks, 4, n_stacks)
    assert np.isfinite(t).all()
    assert (t >= 0).all()


# ============================================================================
# JKO step invariants
# ============================================================================


@given(
    rho=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=8),
        elements=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    ).map(lambda a: a / a.sum()),
    h=st.floats(min_value=0.001, max_value=0.5, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_jko_step_preserves_mass_and_increments_tau(rho: np.ndarray, h: float) -> None:
    """JKOStep.step must preserve mass within tol and increment tau by 1."""
    state = _make_state(rho)
    support = np.linspace(0, 1, rho.size).reshape(-1, 1)
    step = JKOStep(f_functional=ZeroF(), h=h, support=support, n_inner=5)
    new_state = step.step(state)
    new_rho = new_state.rho["phono"]
    assert abs(new_rho.sum() - 1.0) < MASS_TOL
    assert (new_rho >= -NEG_TOL).all()
    assert new_state.tau == state.tau + 1
