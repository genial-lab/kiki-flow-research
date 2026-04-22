"""Fast/slow nested loop orchestration for Track 2 paper runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.particle_simulator import ParticleBatch, ParticleSimulator


def _zero_potential(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def _particles_to_flow_state(
    particles: ParticleBatch,
    support: np.ndarray,
    species_names: list[str],
) -> FlowState:
    """KDE-free histogramming: bin the 1D projection of positions per species.

    Accepts either numpy or MLX positions; converts to numpy for np.histogram.
    """
    n_bins = support.shape[0]
    lo = float(support[0, 0])
    hi = float(support[-1, 0])
    rho: dict[str, np.ndarray] = {}
    tags = np.asarray(particles["species_tags"])
    positions_np = np.asarray(particles["positions"])
    positions_1d = positions_np[:, 0]
    for name in species_names:
        mask = tags == name
        if not mask.any():
            rho[name] = np.full(n_bins, 1.0 / n_bins)
            continue
        hist, _ = np.histogram(positions_1d[mask], bins=n_bins, range=(lo, hi))
        total = hist.sum()
        rho[name] = (hist / total) if total > 0 else np.full(n_bins, 1.0 / n_bins)
    return FlowState(
        rho=rho,
        P_theta=np.zeros(4),
        mu_curr=np.full(n_bins, 1.0 / n_bins),
        tau=0,
        metadata={"track_id": "T2"},
    )


class MultiscaleLoop:
    """Nest N_fast Langevin substeps inside each slow JKO step.

    Accepts any simulator object with ``initialize``/``evolve``/``species``
    attributes, including the numpy ``ParticleSimulator`` and the MLX
    ``MLXParticleSimulator``. For MLX, ``potential_fn`` is left as None
    (zero drift) since the numpy ``_zero_potential`` would not type-match.
    """

    def __init__(
        self,
        sim: ParticleSimulator | Any,
        jko: JKOStep,
        n_fast: int,
        n_slow: int,
        support: np.ndarray,
        dt_fast: float = 1e-3,
        post_jko_hook: Callable[[FlowState], FlowState] | None = None,
    ) -> None:
        self.sim = sim
        self.jko = jko
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.support = support
        self.dt_fast = dt_fast
        self.post_jko_hook = post_jko_hook
        self._is_mlx = type(sim).__name__ == "MLXParticleSimulator"

    def run(self, seed: int) -> dict[str, Any]:
        particles = self.sim.initialize()
        names = self.sim.species.species_names()
        trajectory: list[FlowState] = []
        for _ in range(self.n_slow):
            if self._is_mlx:
                mlx_sim: Any = self.sim
                particles = mlx_sim.evolve(
                    particles,
                    dt=self.dt_fast,
                    n_steps=self.n_fast,
                    potential_fn=None,
                )
            else:
                particles = self.sim.evolve(
                    particles,
                    dt=self.dt_fast,
                    n_steps=self.n_fast,
                    potential_fn=_zero_potential,
                )
            state = _particles_to_flow_state(particles, self.support, names)
            state = self.jko.step(state)
            if self.post_jko_hook is not None:
                state = self.post_jko_hook(state)
            trajectory.append(state)
        return {
            "seed": seed,
            "n_slow_completed": len(trajectory),
            "trajectory": trajectory,
        }
