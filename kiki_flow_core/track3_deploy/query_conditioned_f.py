"""QueryConditionedF — AIF FreeEnergy for text-conditioned Wasserstein flow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.state import FlowState

SPECIES_CANONICAL: tuple[str, ...] = ("phono:code", "sem:code", "lex:code", "syntax:code")
_N_SPECIES = 4
_N_STACKS = 32
_DENSITY_FLOOR = 1e-12
_LOG_EPS = 1e-8
_DEFAULT_LAMBDA_J = 0.1
_DEFAULT_SIGMA2 = 1.0


def _g_jepa_forward(params: dict[str, jnp.ndarray], rho_flat: jnp.ndarray) -> jnp.ndarray:
    """2-layer MLP: rho_flat (128,) → hidden (256,) → embedding (384,)."""
    h = jax.nn.gelu(rho_flat @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


class QueryConditionedF(FreeEnergy):
    """FreeEnergy coupling text embedding to Wasserstein flow via JEPA decoder."""

    def __init__(
        self,
        g_jepa_params: dict[str, np.ndarray],
        embedding: np.ndarray,
        pi_prior: dict[str, np.ndarray] | None = None,
        coupling_matrix: np.ndarray | None = None,
        lambda_j: float = _DEFAULT_LAMBDA_J,
        sigma2: float = _DEFAULT_SIGMA2,
    ) -> None:
        self.g_jepa_params: dict[str, jnp.ndarray] = {
            k: jnp.asarray(v) for k, v in g_jepa_params.items()
        }
        self.embedding: jnp.ndarray = jnp.asarray(embedding, dtype=jnp.float32)
        if pi_prior is None:
            pi_prior = {
                sp: np.ones(_N_STACKS, dtype=np.float32) / _N_STACKS for sp in SPECIES_CANONICAL
            }
        self.pi_prior = pi_prior
        if coupling_matrix is None:
            coupling_matrix = np.zeros((_N_SPECIES, _N_SPECIES), dtype=np.float32)
        self.coupling_matrix = coupling_matrix
        self.lambda_j = lambda_j
        self.sigma2 = sigma2
        # Pre-compile JAX gradient for JEPA accuracy term
        self._grad_jepa_fn = jax.jit(jax.grad(self._jepa_loss))

    def _flatten_rho(self, state: FlowState) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(state.rho[sp]) for sp in SPECIES_CANONICAL])

    def _jepa_loss(self, rho_flat: jnp.ndarray) -> jnp.ndarray:
        pred = _g_jepa_forward(self.g_jepa_params, rho_flat)
        diff = self.embedding - pred
        return 0.5 * jnp.sum(diff**2) / self.sigma2

    def value(self, state: FlowState) -> float:
        # Complexity term
        complexity = 0.0
        for sp in SPECIES_CANONICAL:
            rho = np.clip(state.rho[sp], _DENSITY_FLOOR, None)
            prior = np.clip(self.pi_prior[sp], _DENSITY_FLOOR, None)
            complexity += float(np.sum(rho * (np.log(rho + _LOG_EPS) - np.log(prior + _LOG_EPS))))
        # Accuracy term (JEPA)
        rho_flat = self._flatten_rho(state)
        accuracy = float(self._jepa_loss(rho_flat))
        # Coupling term
        coupling = 0.0
        if self.lambda_j > 0:
            for i, si in enumerate(SPECIES_CANONICAL):
                for j_idx, sj in enumerate(SPECIES_CANONICAL):
                    jij = float(self.coupling_matrix[i, j_idx])
                    if jij != 0.0:
                        coupling += jij * float(np.dot(state.rho[si], state.rho[sj]))
            coupling *= self.lambda_j
        return complexity + accuracy + coupling

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        if species_name not in self.pi_prior:
            raise ValueError(f"Unknown species: {species_name!r}")
        # Complexity gradient: log(rho_s / pi_prior_s) + 1
        rho_s = np.clip(state.rho[species_name], _DENSITY_FLOOR, None)
        prior_s = np.clip(self.pi_prior[species_name], _DENSITY_FLOOR, None)
        grad_complexity = np.log(rho_s + _LOG_EPS) - np.log(prior_s + _LOG_EPS) + 1.0
        # Accuracy gradient: autodiff via jax
        rho_flat = self._flatten_rho(state)
        grad_jepa_flat = np.asarray(self._grad_jepa_fn(rho_flat))
        species_idx = SPECIES_CANONICAL.index(species_name)
        slice_start = species_idx * _N_STACKS
        slice_end = slice_start + _N_STACKS
        grad_accuracy = grad_jepa_flat[slice_start:slice_end]
        # Coupling gradient: 2 * lambda_j * sum_t J_{s,t} * rho_t
        grad_coupling: np.ndarray = np.zeros(_N_STACKS, dtype=np.float32)
        if self.lambda_j > 0:
            for t, st in enumerate(SPECIES_CANONICAL):
                jst = float(self.coupling_matrix[species_idx, t])
                if jst != 0.0:
                    grad_coupling = (grad_coupling + jst * np.asarray(state.rho[st])).astype(
                        np.float32
                    )
            grad_coupling = (grad_coupling * (2.0 * self.lambda_j)).astype(np.float32)
        out: np.ndarray = (grad_complexity + grad_accuracy + grad_coupling).astype(np.float32)
        return out
