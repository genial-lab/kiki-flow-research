"""Audit test: verify JKO oracle output exposes per-species rho distributions."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.state_projection import flatten, unflatten

EXPECTED_SPECIES = {"phono:code", "lex:code", "syntax:code", "sem:code"}
EXPECTED_STACKS_PER_SPECIES = 32


def test_flowstate_rho_is_per_species_dict(sample_flowstate: FlowState) -> None:
    """FlowState.rho must be a dict keyed by the 4 Levelt-Baddeley species."""
    assert isinstance(sample_flowstate.rho, dict)
    assert set(sample_flowstate.rho.keys()) == EXPECTED_SPECIES


def test_flowstate_rho_shapes(sample_flowstate: FlowState) -> None:
    """Each species rho must be a 1D array of length 32."""
    for species, rho in sample_flowstate.rho.items():
        assert isinstance(rho, np.ndarray), f"{species} rho is {type(rho)}"
        assert rho.shape == (EXPECTED_STACKS_PER_SPECIES,), f"{species} shape {rho.shape}"


def test_flatten_roundtrip(sample_flowstate: FlowState) -> None:
    """flatten/unflatten must preserve all species and values."""
    flat = flatten(sample_flowstate)
    assert flat.shape == (len(EXPECTED_SPECIES) * EXPECTED_STACKS_PER_SPECIES,)
    restored = unflatten(flat, sample_flowstate)
    for species in EXPECTED_SPECIES:
        np.testing.assert_allclose(restored.rho[species], sample_flowstate.rho[species], rtol=1e-8)
