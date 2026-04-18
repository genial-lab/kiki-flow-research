"""FreeEnergy abstract base, ZeroF reference, and generic JKOStep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.wasserstein_ops import prox_w2

ProxFn = Callable[..., np.ndarray]


class FreeEnergy(ABC):
    """Abstract free-energy functional on FlowState."""

    @abstractmethod
    def value(self, state: FlowState) -> float:
        """Return the scalar free-energy F[state]."""

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        """Numerical gradient of F w.r.t. rho[species_name].

        Override for analytical gradients when available. Each coordinate is perturbed
        and the distribution is renormalized so the gradient respects the simplex
        constraint (mass conservation of rho).
        """
        rho = state.rho[species_name]
        grad = np.zeros_like(rho)
        f0 = self.value(state)
        for i in range(rho.size):
            perturbed = rho.copy()
            perturbed[i] += eps
            perturbed = perturbed / perturbed.sum()
            new_state = state.model_copy(update={"rho": {**state.rho, species_name: perturbed}})
            grad[i] = (self.value(new_state) - f0) / eps
        return grad


class ZeroF(FreeEnergy):
    """Identically-zero free energy. Used in tests and as no-op placeholder."""

    def value(self, state: FlowState) -> float:
        return 0.0

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        return np.zeros_like(state.rho[species_name])


class JKOStep:
    """Generic JKO scheme step: argmin_{rho} F(rho) + (1/2h) * W2^2(rho, rho_tau).

    v1 operates on the rho marginal only (P_theta and mu_curr held fixed) and
    implements F-gradient descent on the simplex (projected after each substep).
    Track-specific subclasses may add the full Wasserstein regularization term
    via ``wasserstein_ops.prox_w2`` when needed for rigorous JKO semantics.
    """

    def __init__(
        self,
        f_functional: FreeEnergy,
        h: float,
        support: np.ndarray,
        n_inner: int = 50,
        apply_w2_prox: bool = False,
        prox_fn: ProxFn | None = None,
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        self.f_functional = f_functional
        self.h = h
        self.support = support
        self.n_inner = n_inner
        self.apply_w2_prox = apply_w2_prox
        # Strategy pattern: Track-specific subclasses can inject an MLX-backed
        # prox without forcing a circular import here. Default remains the
        # POT-backed prox_w2.
        self.prox_fn: ProxFn = prox_fn if prox_fn is not None else prox_w2  # type: ignore[assignment]

    def step(self, state: FlowState) -> FlowState:
        new_rho: dict[str, np.ndarray] = {}
        substep = self.h / max(self.n_inner, 1)
        for name, rho_tau in state.rho.items():
            if rho_tau.size != self.support.shape[0]:
                new_rho[name] = rho_tau
                continue
            rho = rho_tau.copy()
            for _ in range(self.n_inner):
                step_state = state.model_copy(update={"rho": {**state.rho, name: rho}})
                grad_f = self.f_functional.grad_rho(step_state, name)
                rho = rho - substep * grad_f
                rho = np.clip(rho, 1e-12, None)
                rho = rho / rho.sum()
            if self.apply_w2_prox:
                rho = self.prox_fn(
                    rho,
                    reference=rho_tau,
                    epsilon=self.h,
                    support=self.support,
                    n_iter=100,
                )
            new_rho[name] = rho
        return state.model_copy(update={"rho": new_rho, "tau": state.tau + 1})
