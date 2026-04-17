import numpy as np

from kiki_flow_core.master_equation import ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver


def make_state(rho: np.ndarray) -> FlowState:
    return FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def test_full_jko_applies_w2_prox():
    rho = np.array([0.25, 0.25, 0.25, 0.25])
    state = make_state(rho)
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    solver = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support, epsilon=0.01)
    new_state = solver.step(state)
    np.testing.assert_allclose(new_state.rho["phono"], rho, atol=0.05)  # noqa: PLR2004
    assert solver.apply_w2_prox is True


def test_full_jko_tau_increments():
    support = np.linspace(0, 1, 2).reshape(-1, 1)
    state = make_state(np.array([0.5, 0.5]))
    solver = FullJKOSolver(f_functional=ZeroF(), h=0.05, support=support, epsilon=0.01)
    new_state = solver.step(state)
    assert new_state.tau == state.tau + 1
