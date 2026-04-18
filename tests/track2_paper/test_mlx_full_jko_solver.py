import numpy as np

from kiki_flow_core.master_equation import ZeroF
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver, MLXFullJKOSolver
from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_prox_w2


def _make_state(rho: np.ndarray) -> FlowState:
    return FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(4),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T2"},
    )


def test_mlx_full_jko_solver_constructs_with_mlx_prox():
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    solver = MLXFullJKOSolver(f_functional=ZeroF(), h=0.05, support=support, epsilon=0.05)
    assert solver.apply_w2_prox is True
    assert solver.epsilon == 0.05  # noqa: PLR2004
    # prox_fn is the MLX one, not the POT default
    assert solver.prox_fn is mlx_prox_w2


def test_mlx_full_jko_solver_step_increments_tau():
    support = np.linspace(0, 1, 4).reshape(-1, 1)
    state = _make_state(np.full(4, 0.25))
    solver = MLXFullJKOSolver(f_functional=ZeroF(), h=0.05, support=support, epsilon=0.1, n_inner=5)
    new_state = solver.step(state)
    assert new_state.tau == state.tau + 1


def test_full_jko_solver_epsilon_and_max_iter_persist():
    support = np.linspace(0, 1, 2).reshape(-1, 1)
    solver = FullJKOSolver(f_functional=ZeroF(), h=0.1, support=support, epsilon=0.02, max_iter=500)
    assert solver.epsilon == 0.02  # noqa: PLR2004
    assert solver.max_iter == 500  # noqa: PLR2004
