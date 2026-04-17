"""Top-level driver: run 5 seeds, produce figures, aggregate stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.species import OrthoSpecies
from kiki_flow_core.track2_paper.figures.phase_portrait import make_phase_portrait
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator


def run_paper(
    seeds: list[int],
    n_particles: int = 10_000,
    n_fast: int = 1000,
    n_slow: int = 100,
    grid_size: int = 64,
    out_dir: Path = Path("paper"),
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    species = OrthoSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, grid_size).reshape(-1, 1)
    potentials = {n: np.zeros(grid_size) for n in names}
    prior = {n: np.full(grid_size, 1.0 / grid_size) for n in names}
    f_func = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)
    jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.01)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        sim = ParticleSimulator(species=species, n_particles=n_particles, latent_dim=2, seed=seed)
        loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=n_fast, n_slow=n_slow, support=support)
        manifest = loop.run(seed=seed)
        make_phase_portrait(
            manifest["trajectory"],
            out_dir / "figures",
            filename=f"fig1_phase_seed{seed}",
        )
        per_seed.append({"seed": seed, "n_slow_completed": manifest["n_slow_completed"]})

    stats = {
        "n_seeds": len(seeds),
        "per_seed": per_seed,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
