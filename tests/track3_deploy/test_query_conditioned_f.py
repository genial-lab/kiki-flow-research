"""Tests for QueryConditionedF — AIF FreeEnergy for text-conditioned Wasserstein flow."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.query_conditioned_f import QueryConditionedF

N_SPECIES = 4
N_STACKS = 32
EMBED_DIM = 384
HIDDEN_DIM = 256
GRAD_TOL = 1e-3
LAMBDA_J_DEFAULT = 0.1
SIGMA2_DEFAULT = 1.0
WEIGHT_INIT_SCALE = 0.01
P_THETA_DIM = 16


def _uniform_state() -> FlowState:
    rho = {
        f"{s}:code": np.ones(N_STACKS, dtype=np.float32) / N_STACKS
        for s in ("phono", "sem", "lex", "syntax")
    }
    return FlowState(
        rho=rho,
        P_theta=np.zeros(P_THETA_DIM, dtype=np.float32),
        mu_curr=np.zeros(1, dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )


def _random_g_jepa_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.standard_normal((N_SPECIES * N_STACKS, HIDDEN_DIM)) * WEIGHT_INIT_SCALE).astype(
            np.float32
        ),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": (rng.standard_normal((HIDDEN_DIM, EMBED_DIM)) * WEIGHT_INIT_SCALE).astype(np.float32),
        "b2": np.zeros(EMBED_DIM, dtype=np.float32),
    }


def _zero_g_jepa_params() -> dict[str, np.ndarray]:
    return {
        "W1": np.zeros((N_SPECIES * N_STACKS, HIDDEN_DIM), dtype=np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": np.zeros((HIDDEN_DIM, EMBED_DIM), dtype=np.float32),
        "b2": np.zeros(EMBED_DIM, dtype=np.float32),
    }


def _random_embedding(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)


def test_value_is_finite() -> None:
    state = _uniform_state()
    f_energy = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=_random_embedding(),
        lambda_j=LAMBDA_J_DEFAULT,
        sigma2=SIGMA2_DEFAULT,
    )
    v = f_energy.value(state)
    assert np.isfinite(v)


def test_grad_rho_shape() -> None:
    state = _uniform_state()
    f_energy = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=_random_embedding(),
    )
    for species in state.rho:
        grad = f_energy.grad_rho(state, species)
        assert grad.shape == (N_STACKS,)


def test_limit_g_jepa_zero_reduces_to_complexity() -> None:
    """If g_JEPA weights are zero, lambda_j=0, uniform rho with uniform prior → F ≈ 0."""
    state = _uniform_state()
    f_energy = QueryConditionedF(
        g_jepa_params=_zero_g_jepa_params(),
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
        sigma2=SIGMA2_DEFAULT,
    )
    assert abs(f_energy.value(state)) < GRAD_TOL


def test_grad_at_uniform_with_uniform_prior_is_near_zero() -> None:
    """At uniform rho with uniform prior, complexity gradient is uniform (constant)."""
    state = _uniform_state()
    f_energy = QueryConditionedF(
        g_jepa_params=_zero_g_jepa_params(),
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
    )
    grad = f_energy.grad_rho(state, "phono:code", eps=1e-4)
    # All entries equal (gradient is constant for uniform input)
    assert np.allclose(grad, grad[0], atol=GRAD_TOL)


def test_value_strictly_positive_when_rho_peaked() -> None:
    """F.value > 0 when rho is peaked (KL to uniform > 0)."""
    rho = {
        f"{s}:code": np.eye(N_STACKS)[0].astype(np.float32)
        for s in ("phono", "sem", "lex", "syntax")
    }
    # Add tiny epsilon to avoid log(0)
    for k, val in rho.items():
        rho[k] = val + 1e-6
        rho[k] /= rho[k].sum()
    state = FlowState(
        rho=rho,
        P_theta=np.zeros(P_THETA_DIM, dtype=np.float32),
        mu_curr=np.zeros(1, dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )
    f_energy = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
    )
    assert f_energy.value(state) > 0
