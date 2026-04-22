"""Separable baseline free energy for the Track 2 paper ablation study.

F_sep(rho) = sum_i <rho_i, V_i> + sum_i KL(rho_i || prior_i)

No inter-species coupling (J = 0) and no Turing cross-diffusion. Used to
quantify how much specialization is attributable to the hierarchical J
coupling versus the per-species potential/entropy machinery alone.
"""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState


class SeparableEnergy(FreeEnergy):
    """F_sep = sum_i <rho_i, V_i> + sum_i KL(rho_i || prior_i). Zero J, zero Turing."""

    def __init__(
        self,
        species: CanonicalSpecies,
        potentials: dict[str, np.ndarray],
        prior: dict[str, np.ndarray],
    ) -> None:
        self.species = species
        self.potentials = potentials
        self.prior = prior

    def value(self, state: FlowState) -> float:
        total = 0.0
        for n in self.species.species_names():
            rho = state.rho[n]
            total += float(np.dot(rho, self.potentials[n]))
            rho_safe = np.clip(rho, 1e-12, None)
            prior_safe = np.clip(self.prior[n], 1e-12, None)
            total += float(np.sum(rho_safe * np.log(rho_safe / prior_safe)))
        return total
