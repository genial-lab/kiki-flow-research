"""T1 phenomenological free energy: attraction + entropy + reaction coupling."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState


class T1FreeEnergy(FreeEnergy):
    """F_T1[rho] = alpha * <rho, V_curr> + beta * entropy(rho) + gamma * reaction_overlap."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        species: MixedCanonicalSpecies,
        v_curr: np.ndarray,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.species = species
        self.v_curr = v_curr
        self._coupling_2d = species.coupling_matrix()

    def value(self, state: FlowState) -> float:
        v = self.v_curr
        total = 0.0
        names = self.species.species_names()
        rhos = [state.rho[n] for n in names]

        for r in rhos:
            total += self.alpha * float(np.dot(r, v))
            r_safe = np.clip(r, 1e-12, None)
            total += self.beta * float(-(r_safe * np.log(r_safe)).sum())

        if self.gamma != 0.0:
            reaction = 0.0
            j = self._coupling_2d
            for i, ri in enumerate(rhos):
                for k, rk in enumerate(rhos):
                    reaction += float(j[i, k] * np.dot(ri, rk))
            total += self.gamma * reaction

        return total

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        v = self.v_curr
        rho = state.rho[species_name]
        rho_safe = np.clip(rho, 1e-12, None)
        grad_attr = self.alpha * v
        grad_ent = -self.beta * (np.log(rho_safe) + 1.0)

        if self.gamma != 0.0:
            names = self.species.species_names()
            idx = names.index(species_name)
            j = self._coupling_2d
            grad_reaction = np.zeros_like(rho)
            for k, name_k in enumerate(names):
                grad_reaction += j[idx, k] * state.rho[name_k]
            grad_reaction = 2.0 * self.gamma * grad_reaction
        else:
            grad_reaction = np.zeros_like(rho)

        out: np.ndarray = grad_attr + grad_ent + grad_reaction
        return out
