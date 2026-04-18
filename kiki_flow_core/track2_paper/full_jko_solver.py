"""T2 JKO solvers: POT-backed (default) and MLX-backed variants."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep
from kiki_flow_core.track2_paper.mlx_wasserstein import mlx_prox_w2


class FullJKOSolver(JKOStep):
    """JKOStep with apply_w2_prox=True and tighter Sinkhorn params for paper rigor."""

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        n_inner: int = 50,
    ) -> None:
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
        )
        self.epsilon = epsilon
        self.max_iter = max_iter


class MLXFullJKOSolver(JKOStep):
    """JKOStep with Wasserstein prox routed through the native MLX Sinkhorn.

    Mirrors ``FullJKOSolver`` but uses ``mlx_prox_w2`` for the proximal
    step. Expected to be 5 to 10 times faster on Apple Silicon for the
    rigorous path; falls back silently if MLX is unavailable by routing
    through the POT default.
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        n_inner: int = 50,
    ) -> None:
        super().__init__(
            f_functional=f_functional,
            h=h,
            support=support,
            n_inner=n_inner,
            apply_w2_prox=True,
            prox_fn=mlx_prox_w2,
        )
        self.epsilon = epsilon
        self.max_iter = max_iter
