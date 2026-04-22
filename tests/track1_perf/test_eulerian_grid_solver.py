import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.modules import AdvectionDiffusion, PhonologicalLoop, ScaffoldingScheduler
from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState, assert_invariants
from kiki_flow_core.track1_perf.eulerian_grid_solver import EulerianGridSolver
from kiki_flow_core.track1_perf.phenomenological_f import T1FreeEnergy


def make_initial(species: MixedCanonicalSpecies, n_grid: int = 16) -> FlowState:
    names = species.species_names()
    x = np.linspace(-1, 1, n_grid)
    rho = np.exp(-0.5 * (x / 0.3) ** 2)
    rho /= rho.sum()
    return FlowState(
        rho={name: rho.copy() for name in names},
        P_theta=np.zeros(4),
        mu_curr=np.full(n_grid, 1.0 / n_grid),
        tau=0,
        metadata={"track_id": "T1"},
    )


def test_eulerian_step_increments_tau_and_preserves_invariants():
    species = MixedCanonicalSpecies(stack_names=["code"])
    state = make_initial(species)
    x = np.linspace(-1, 1, 16)
    adv_diff = AdvectionDiffusion(species=species, x_grid=x, diffusion=0.001)
    scheduler = ScaffoldingScheduler(h_min=1e-2, h_max=0.1)
    phono = PhonologicalLoop(detector=np.zeros_like, correction_strength=0.0)
    f = T1FreeEnergy(alpha=0.0, beta=0.1, gamma=0.0, species=species, v_curr=np.zeros(16))
    jko = JKOStep(f_functional=f, h=0.05, support=x.reshape(-1, 1), n_inner=5)
    solver = EulerianGridSolver(species, scheduler, adv_diff, jko, phono)
    new_state = solver.step(state)
    assert new_state.tau == state.tau + 1
    assert_invariants(new_state)
