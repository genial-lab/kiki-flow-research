"""Tests for SeparableEnergy (ablation baseline: J = 0, Turing = 0)."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.paper_f_separable import SeparableEnergy

N_GRID = 8
SPECIES_NAMES = ("phono", "lex", "syntax", "sem")
EXACT_TOL = 1e-12


def _random_simplex(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw a Dirichlet(1) sample -- uniform over the (n-1)-simplex."""
    return rng.dirichlet(np.ones(n))


def _make_state(rng: np.random.Generator, n: int = N_GRID) -> FlowState:
    rhos = {name: _random_simplex(rng, n) for name in SPECIES_NAMES}
    return FlowState(
        rho=rhos,
        P_theta=np.zeros(4),
        mu_curr=np.full(n, 1.0 / n),
        tau=0,
        metadata={"track_id": "T2"},
    )


def _random_potentials(rng: np.random.Generator, n: int = N_GRID) -> dict[str, np.ndarray]:
    return {name: rng.standard_normal(n) for name in SPECIES_NAMES}


def _random_prior(rng: np.random.Generator, n: int = N_GRID) -> dict[str, np.ndarray]:
    return {name: _random_simplex(rng, n) for name in SPECIES_NAMES}


def test_separable_equals_t2_when_j_and_turing_zero() -> None:
    """With J = 0 and turing_strength = 0, T2FreeEnergy collapses to SeparableEnergy."""
    rng = np.random.default_rng(42)
    species = CanonicalSpecies()
    potentials = _random_potentials(rng)
    prior = _random_prior(rng)
    state = _make_state(rng)

    sep = SeparableEnergy(species=species, potentials=potentials, prior=prior)
    t2 = T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.0,
    )
    # Force coupling to zero explicitly: the full coupling, plus the canonical
    # symmetric/antisymmetric split, all need to be identically zero so that
    # the bilinear sum contributes nothing to T2FreeEnergy.value.
    k = len(species.species_names())
    zero_j = np.zeros((k, k))
    t2._coupling = zero_j
    t2._J_sym = zero_j
    t2._J_asym = zero_j

    v_sep = sep.value(state)
    v_t2 = t2.value(state)
    assert np.isfinite(v_sep)
    assert np.isfinite(v_t2)
    assert abs(v_sep - v_t2) < EXACT_TOL


def test_separable_independent_of_second_species() -> None:
    """Perturbing rho_lex changes F_sep exactly by the lex V + KL delta.

    Because F_sep has no cross-species coupling, the change in F when only
    one species' rho is modified must equal the change in that species'
    own <rho, V> + KL contribution -- the other three species' terms
    are bit-exactly invariant.
    """
    rng = np.random.default_rng(7)
    species = CanonicalSpecies()
    potentials = _random_potentials(rng)
    prior = _random_prior(rng)
    state = _make_state(rng)

    sep = SeparableEnergy(species=species, potentials=potentials, prior=prior)

    def lex_contribution(rho_lex: np.ndarray) -> float:
        v = float(np.dot(rho_lex, potentials["lex"]))
        rho_safe = np.clip(rho_lex, 1e-12, None)
        prior_safe = np.clip(prior["lex"], 1e-12, None)
        v += float(np.sum(rho_safe * np.log(rho_safe / prior_safe)))
        return v

    f_before = sep.value(state)
    lex_before = lex_contribution(state.rho["lex"])

    # Perturb rho_lex with a fresh simplex draw.
    new_lex = _random_simplex(rng, N_GRID)
    new_rho = {**state.rho, "lex": new_lex}
    new_state = state.model_copy(update={"rho": new_rho})

    f_after = sep.value(new_state)
    lex_after = lex_contribution(new_lex)

    delta_total = f_after - f_before
    delta_lex = lex_after - lex_before
    assert abs(delta_total - delta_lex) < EXACT_TOL


def test_separable_kl_is_nonnegative() -> None:
    """With prior = uniform and varying rho, the KL piece is always >= 0 (Gibbs).

    We zero out the potentials so F_sep reduces purely to sum_i KL(rho_i || uniform),
    which must be non-negative for every valid state by Gibbs' inequality, with
    equality iff every rho_i equals the uniform prior.
    """
    rng = np.random.default_rng(123)
    species = CanonicalSpecies()
    uniform = np.full(N_GRID, 1.0 / N_GRID)
    potentials = {name: np.zeros(N_GRID) for name in SPECIES_NAMES}
    prior = {name: uniform.copy() for name in SPECIES_NAMES}
    sep = SeparableEnergy(species=species, potentials=potentials, prior=prior)

    # Equality case: rho == uniform everywhere.
    flat_state = FlowState(
        rho={name: uniform.copy() for name in SPECIES_NAMES},
        P_theta=np.zeros(4),
        mu_curr=uniform.copy(),
        tau=0,
        metadata={"track_id": "T2"},
    )
    assert abs(sep.value(flat_state)) < EXACT_TOL

    # Strict-positive case: random simplex rhos, KL > 0.
    for _ in range(5):
        state = _make_state(rng)
        v = sep.value(state)
        assert np.isfinite(v)
        assert v >= -EXACT_TOL
        # Exceedingly unlikely that a Dirichlet(1) draw lands on uniform,
        # so the sum of four KLs should be strictly positive.
        assert v > 0.0
