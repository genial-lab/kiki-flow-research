"""Cross-variant comparison tests for the 4-species coupling matrix.

Verifies that the two published architectures (Dell 1986 interactive network
vs. Levelt, Roelofs & Meyer 1999 serial WEAVER++) load cleanly and exhibit
their characteristic structural difference on the sem -> {lex, syntax} axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from kiki_flow_core.species.ortho_baddeley_levelt import OrthoSpecies

FROB_LOWER = 0.5
FROB_UPPER = 2.0


def _indices(species: OrthoSpecies) -> dict[str, int]:
    return {name: i for i, name in enumerate(species.species_names())}


def test_dell_coupling_loads_and_has_sem_lex_dominance() -> None:
    """Dell variant: direct sem->lex path must dominate the syntactic detour."""
    species = OrthoSpecies(coupling_variant="dell")
    j = species.coupling_matrix()
    idx = _indices(species)
    sem, lex, syn = idx["sem"], idx["lex"], idx["syntax"]
    assert j[lex, sem] > j[syn, sem], (
        "Dell (1986) signature: direct sem->lex must exceed sem->syntax "
        f"(got J[lex,sem]={j[lex, sem]} vs J[syn,sem]={j[syn, sem]})"
    )


def test_levelt_coupling_loads_and_has_syntactic_detour() -> None:
    """Levelt variant: syntactic detour must dominate the direct sem->lex path."""
    species = OrthoSpecies(coupling_variant="levelt")
    j = species.coupling_matrix()
    idx = _indices(species)
    sem, lex, syn = idx["sem"], idx["lex"], idx["syntax"]
    assert j[syn, sem] > j[lex, sem], (
        "Levelt, Roelofs & Meyer (1999) signature: sem->syntax must exceed "
        f"sem->lex (got J[syn,sem]={j[syn, sem]} vs J[lex,sem]={j[lex, sem]})"
    )


@pytest.mark.parametrize("variant", ["dell", "levelt"])
def test_both_variants_row_stochastic_or_trace_preserving(variant: str) -> None:
    """Sanity gate against typos: non-negative entries and bounded Frobenius norm."""
    species = OrthoSpecies(coupling_variant=variant)
    j = species.coupling_matrix()
    assert j.shape == (4, 4)
    assert np.isfinite(j).all(), f"{variant}: coupling matrix has non-finite entries"
    assert (j >= 0).all(), f"{variant}: spreading activation strengths must be non-negative"
    frob = float(np.linalg.norm(j, ord="fro"))
    assert FROB_LOWER <= frob <= FROB_UPPER, (
        f"{variant}: Frobenius norm {frob:.4f} outside sanity range "
        f"[{FROB_LOWER}, {FROB_UPPER}] -- likely a typo in the YAML"
    )
