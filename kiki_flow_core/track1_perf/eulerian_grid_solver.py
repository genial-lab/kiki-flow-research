"""Orchestrates one T1 consolidation step over the full MixedCanonicalSpecies state tensor."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.modules import AdvectionDiffusion, PhonologicalLoop, ScaffoldingScheduler
from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState


class EulerianGridSolver:
    """Applies scheduler -> phono loop -> advection-diffusion -> JKO in sequence."""

    def __init__(
        self,
        species: MixedCanonicalSpecies,
        scheduler: ScaffoldingScheduler,
        adv_diff: AdvectionDiffusion,
        jko: JKOStep,
        phono: PhonologicalLoop,
    ) -> None:
        self.species = species
        self.scheduler = scheduler
        self.adv_diff = adv_diff
        self.jko = jko
        self.phono = phono

    def step(self, state: FlowState, error_profile: np.ndarray | None = None) -> FlowState:
        names = self.species.species_names()
        if error_profile is None:
            error_profile = np.full(len(names), 0.1)
        h, _mu_curr = self.scheduler.next_step(error_profile=error_profile)
        first_rho = next(iter(state.rho.values()))
        v_field = np.zeros_like(first_rho)

        new_rho: dict[str, np.ndarray] = {}
        for name, rho in state.rho.items():
            source = self.phono.source_term(rho_phono=rho, output=rho) if "phono" in name else None
            new_rho[name] = self.adv_diff.step_1d(rho, v_field=v_field, dt=h, source=source)

        state_intermediate = state.model_copy(update={"rho": new_rho})
        return self.jko.step(state_intermediate)
