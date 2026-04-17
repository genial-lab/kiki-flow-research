"""T2 JKO solver with mandatory Wasserstein proximal term via POT Sinkhorn."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy, JKOStep


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
