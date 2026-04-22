import numpy as np
import pytest

from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track1_perf.phenomenological_f import T1FreeEnergy


def make_state(stacks: list[str], n_grid: int = 8) -> tuple[FlowState, MixedCanonicalSpecies]:
    species = MixedCanonicalSpecies(stack_names=stacks)
    names = species.species_names()
    rho = {name: np.full(n_grid, 1.0 / n_grid) for name in names}
    return (
        FlowState(
            rho=rho,
            P_theta=np.zeros(8),
            mu_curr=np.full(n_grid, 1.0 / n_grid),
            tau=0,
            metadata={"track_id": "T1"},
        ),
        species,
    )


def test_t1_free_energy_finite_on_uniform_state():
    state, species = make_state(["code", "math"])
    f = T1FreeEnergy(alpha=1.0, beta=0.1, gamma=0.5, species=species, v_curr=np.zeros(8))
    assert np.isfinite(f.value(state))


def test_t1_free_energy_flows_downhill_in_potential():
    """Physics convention: F = <rho, V> decreases when mass concentrates where V is low."""
    state, species = make_state(["code"])
    v_linear = np.linspace(0.0, 1.0, 8)
    f = T1FreeEnergy(alpha=1.0, beta=0.0, gamma=0.0, species=species, v_curr=v_linear)
    shifted = state.model_copy(
        update={"rho": {name: np.array([0.2] * 4 + [0.05] * 4) for name in state.rho}}
    )
    assert f.value(shifted) < f.value(state)


def test_t1_free_energy_gradient_shape():
    state, species = make_state(["code"])
    f = T1FreeEnergy(alpha=1.0, beta=0.1, gamma=0.5, species=species, v_curr=np.zeros(8))
    for name in state.rho:
        grad = f.grad_rho(state, name)
        assert grad.shape == state.rho[name].shape


def test_t1_free_energy_gamma_zero_differs_from_gamma_positive():
    state, species = make_state(["code", "math"])
    f_no = T1FreeEnergy(alpha=1.0, beta=0.1, gamma=0.0, species=species, v_curr=np.zeros(8))
    f_yes = T1FreeEnergy(alpha=1.0, beta=0.1, gamma=0.5, species=species, v_curr=np.zeros(8))
    assert f_yes.value(state) != pytest.approx(f_no.value(state))
