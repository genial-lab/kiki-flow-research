"""Regression tests for T2FreeEnergy.apply_drift_splitting.

Covers:
- Identity on uniform states (M @ uniform = uniform after renormalization).
- Per-species mass preservation (simplex constraint kept).
- Non-identity action on a random positive non-uniform state.
- Identity fallback when J_asym is zeroed (symmetric modes).
"""

from __future__ import annotations

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
MASS_TOL = 1e-10
IDENTITY_TOL = 1e-12
NON_IDENTITY_MIN = 1e-6


def _make_energy(variant: str = "dell") -> T2FreeEnergy:
    species = CanonicalSpecies(coupling_variant=variant)
    names = species.species_names()
    potentials = {n: np.zeros(GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    return T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.0,
    )


def _make_state(rho_dict: dict[str, np.ndarray]) -> FlowState:
    first = next(iter(rho_dict.values()))
    return FlowState(
        rho=rho_dict,
        P_theta=np.zeros(4),
        mu_curr=np.full_like(first, 1.0 / first.size),
        tau=0,
        metadata={"track_id": "T2"},
    )


def test_drift_splitting_identity_on_uniform_state() -> None:
    """Uniform rho for every species is a fixed point of the drift step.

    For the orthogonal rotation M, M @ (1, 1, 1, 1)^T = (c1, c2, c3, c4)^T; after
    renormalisation each species is rescaled back to total mass 1 and
    redistributed uniformly over the grid, which equals the input.
    """
    energy = _make_energy()
    names = energy.species.species_names()
    uniform = np.full(GRID, 1.0 / GRID)
    state = _make_state({n: uniform.copy() for n in names})
    new_state = energy.apply_drift_splitting(state, h_drift=0.01)
    for n in names:
        np.testing.assert_allclose(new_state.rho[n], uniform, atol=IDENTITY_TOL)
    assert_invariants(new_state)


def test_drift_splitting_preserves_per_species_mass() -> None:
    """Each rho_i still sums to its prior mass (=1) after the drift step."""
    rng = np.random.default_rng(42)
    energy = _make_energy()
    names = energy.species.species_names()
    rhos = {}
    for n in names:
        raw = rng.random(GRID) + 0.1
        rhos[n] = raw / raw.sum()
    state = _make_state(rhos)
    new_state = energy.apply_drift_splitting(state, h_drift=0.01)
    for n in names:
        mass = float(new_state.rho[n].sum())
        assert abs(mass - 1.0) < MASS_TOL
    assert_invariants(new_state)


def test_drift_splitting_nonidentity_on_nonuniform_state() -> None:
    """A non-uniform positive state should actually change under the step."""
    rng = np.random.default_rng(7)
    energy = _make_energy()
    names = energy.species.species_names()
    rhos = {}
    for n in names:
        raw = rng.random(GRID) + 0.1
        rhos[n] = raw / raw.sum()
    state = _make_state({n: v.copy() for n, v in rhos.items()})
    new_state = energy.apply_drift_splitting(state, h_drift=0.01)
    total_diff = sum(float(np.abs(new_state.rho[n] - rhos[n]).sum()) for n in names)
    assert total_diff > NON_IDENTITY_MIN


def test_drift_splitting_is_identity_when_j_asym_zero() -> None:
    """Zeroing J_asym (symmetric modes) collapses the drift to the identity map."""
    rng = np.random.default_rng(123)
    energy = _make_energy()
    energy._J_asym = np.zeros_like(energy._J_asym)
    names = energy.species.species_names()
    rhos = {}
    for n in names:
        raw = rng.random(GRID) + 0.1
        rhos[n] = raw / raw.sum()
    state = _make_state({n: v.copy() for n, v in rhos.items()})
    new_state = energy.apply_drift_splitting(state, h_drift=0.01)
    for n in names:
        np.testing.assert_allclose(new_state.rho[n], rhos[n], atol=IDENTITY_TOL)


def test_drift_splitting_levelt_variant_runs() -> None:
    """Smoke test: the splitting works for the Levelt coupling variant too."""
    rng = np.random.default_rng(5)
    energy = _make_energy(variant="levelt")
    names = energy.species.species_names()
    rhos = {}
    for n in names:
        raw = rng.random(GRID) + 0.1
        rhos[n] = raw / raw.sum()
    state = _make_state(rhos)
    new_state = energy.apply_drift_splitting(state, h_drift=0.01)
    assert_invariants(new_state)
