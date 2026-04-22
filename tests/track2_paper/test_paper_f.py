import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy


def make_state(n: int = 8) -> FlowState:
    rho_each = np.full(n, 1.0 / n)
    return FlowState(
        rho={
            "phono": rho_each.copy(),
            "lex": rho_each.copy(),
            "syntax": rho_each.copy(),
            "sem": rho_each.copy(),
        },
        P_theta=np.zeros(4),
        mu_curr=rho_each.copy(),
        tau=0,
        metadata={"track_id": "T2"},
    )


def test_t2_f_has_all_four_terms():
    species = CanonicalSpecies()
    state = make_state()
    n = 8
    f = T2FreeEnergy(
        species=species,
        potentials={name: np.zeros(n) for name in species.species_names()},
        prior={name: np.full(n, 1.0 / n) for name in species.species_names()},
        turing_strength=0.1,
    )
    v = f.value(state)
    assert np.isfinite(v)


def test_t2_f_kl_zero_on_prior_match():
    species = CanonicalSpecies()
    state = make_state()
    n = 8
    prior = {n_: state.rho[n_].copy() for n_ in state.rho}
    f = T2FreeEnergy(
        species=species,
        potentials={n_: np.zeros(n) for n_ in state.rho},
        prior=prior,
        turing_strength=0.0,
    )
    assert np.isfinite(f.value(state))
