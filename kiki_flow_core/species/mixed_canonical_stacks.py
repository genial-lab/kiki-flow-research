"""Mixed canonical species: 4 canonical x N-stacks with row-stochastic projection."""

from __future__ import annotations

from typing import Literal

import numpy as np

from kiki_flow_core.species.base import SpeciesBase
from kiki_flow_core.species.canonical_species import CanonicalSpecies

ProjectionInit = Literal["identity", "random", "uniform"]


class MixedCanonicalSpecies(SpeciesBase):
    """Mixture of N LoRA stacks projected into the four canonical species.

    Uses a row-stochastic matrix P[canonical, stack] so that each canonical
    species is a convex combination of stacks. Not an orthogonal
    decomposition.
    """

    def __init__(
        self,
        stack_names: list[str],
        projection_init: ProjectionInit = "uniform",
        seed: int = 0,
    ) -> None:
        if not stack_names:
            raise ValueError("stack_names must be non-empty")
        self._stack_names = list(stack_names)
        self._canonical = CanonicalSpecies()
        self._n_stacks = len(stack_names)
        self._projection = self._init_projection(projection_init, seed)

    def _init_projection(self, kind: ProjectionInit, seed: int) -> np.ndarray:
        n_c, n_s = 4, self._n_stacks
        if kind == "identity":
            p = np.zeros((n_c, n_s))
            for i in range(min(n_c, n_s)):
                p[i, i] = 1.0
            return p
        if kind == "uniform":
            return np.full((n_c, n_s), 1.0 / n_s)
        if kind == "random":
            rng = np.random.default_rng(seed)
            p = rng.random((n_c, n_s))
            p /= p.sum(axis=1, keepdims=True)
            return p
        raise ValueError(f"Unknown projection_init: {kind}")

    def species_names(self) -> list[str]:
        return [f"{c}:{s}" for c in self._canonical.species_names() for s in self._stack_names]

    def projection_matrix(self) -> np.ndarray:
        return self._projection.copy()

    def coupling_tensor(self) -> np.ndarray:
        """4D coupling J[i_c, i_s, j_c, j_s] = J_canonical[i,j] * P[i,i_s] * P[j,j_s]."""
        j_c = self._canonical.coupling_matrix()
        p = self._projection
        out: np.ndarray = np.einsum("ij,is,jt->isjt", j_c, p, p)
        return out

    def coupling_matrix(self) -> np.ndarray:
        """2D flattened view of the coupling tensor.

        For single-stack uniform projection (P row-sums = 1), reduces to
        CanonicalSpecies J for convenient compatibility with pure-canonical
        consumers.
        """
        if self._n_stacks == 1 and np.allclose(self._projection.sum(axis=1), 1.0):
            return self._canonical.coupling_matrix()
        j = self.coupling_tensor()
        n = 4 * self._n_stacks
        return j.reshape(n, n)
