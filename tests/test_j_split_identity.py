"""Regression test: J_sym + J_asym split is bit-for-bit identical to the full J.

The T2FreeEnergy class splits its Levelt-Baddeley coupling J into a symmetric
part J_sym (conservative gradient drive) and an antisymmetric part J_asym
(non-conservative drift). By definition (J_sym + J_asym) rho = J rho for any
rho stack; the helpers on T2FreeEnergy must realise this identity at machine
precision, otherwise the refactor has silently changed dynamics.
"""

from __future__ import annotations

import numpy as np

from kiki_flow_core.species import OrthoSpecies
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

MACHINE_ATOL = 1e-14


def _random_simplex_rhos(n_species: int, grid: int, seed: int) -> list[np.ndarray]:
    """Draw ``n_species`` positive, normalised density vectors of length ``grid``."""
    rng = np.random.default_rng(seed)
    rhos: list[np.ndarray] = []
    for _ in range(n_species):
        raw = rng.random(grid) + 1e-6  # strictly positive
        rhos.append(raw / raw.sum())
    return rhos


def _instantiate_default_f(grid: int) -> T2FreeEnergy:
    species = OrthoSpecies()
    names = species.species_names()
    potentials = {n: np.zeros(grid) for n in names}
    prior = {n: np.full(grid, 1.0 / grid) for n in names}
    return T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=0.0,
    )


def test_j_sym_plus_j_asym_equals_j_identity() -> None:
    """Sanity: J_sym + J_asym reconstructs J exactly (no rounding)."""
    f = _instantiate_default_f(grid=8)
    reconstructed = f._J_sym + f._J_asym
    assert np.allclose(f._coupling, reconstructed, atol=MACHINE_ATOL, rtol=0.0)


def test_grad_conservative_plus_drift_equals_full_j_drive() -> None:
    """Per-species (J_sym rho)_i + (J_asym rho)_i == (J rho)_i, machine precision."""
    grid = 12
    f = _instantiate_default_f(grid)
    rhos = _random_simplex_rhos(n_species=4, grid=grid, seed=20260422)

    # Compute the "legacy" full-J drive directly from the untouched self._coupling
    # matrix so the assertion cannot be satisfied by accident if the split is wrong.
    j_full = f._coupling
    grad_legacy = []
    for i in range(len(rhos)):
        acc = np.zeros_like(rhos[i])
        for k, rk in enumerate(rhos):
            acc = acc + j_full[i, k] * rk
        grad_legacy.append(acc)

    cons = f._grad_conservative(rhos)
    drift = f._drift_nonconservative(rhos)
    grad_new = [c + d for c, d in zip(cons, drift, strict=True)]

    assert len(grad_new) == len(grad_legacy)
    for legacy_i, new_i in zip(grad_legacy, grad_new, strict=True):
        assert np.allclose(legacy_i, new_i, atol=MACHINE_ATOL, rtol=0.0)


def test_coupling_drive_matches_legacy_full_j_drive() -> None:
    """Public legacy helper ``coupling_drive`` reproduces J @ rho element-wise."""
    grid = 10
    f = _instantiate_default_f(grid)
    rhos = _random_simplex_rhos(n_species=4, grid=grid, seed=7)

    drive = f.coupling_drive(rhos)
    for i in range(len(rhos)):
        expected = sum(f._coupling[i, k] * rhos[k] for k in range(len(rhos)))
        assert np.allclose(drive[i], expected, atol=MACHINE_ATOL, rtol=0.0)


def test_value_invariant_under_j_routing_through_sym() -> None:
    """Scalar energy is unchanged: <J_asym, outer(rho, rho)> = 0 identically.

    Guards the refactor's key claim that routing ``value`` through J_sym
    instead of the full J leaves the scalar free-energy output untouched.
    """
    grid = 9
    f = _instantiate_default_f(grid)
    rhos = _random_simplex_rhos(n_species=4, grid=grid, seed=1234)

    # Antisymmetric J contracted with the symmetric rank-2 form sum_ik <rho_i, rho_k>
    # must integrate to zero at machine precision.
    j_asym = f._J_asym
    asym_energy = 0.0
    for i, ri in enumerate(rhos):
        for k, rk in enumerate(rhos):
            asym_energy += float(j_asym[i, k] * np.dot(ri, rk))
    assert abs(asym_energy) < MACHINE_ATOL
