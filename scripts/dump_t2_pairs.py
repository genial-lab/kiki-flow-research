"""Produce (pre, post) rho pairs for T3 distillation by stepping the T2 loop in-process."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.multiscale_loop import _particles_to_flow_state
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator


def _zero(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


STATE_DIM = 16
N_PAIRS = 100
SEED = 0

species = CanonicalSpecies()
names = species.species_names()
support = np.linspace(-2.0, 2.0, STATE_DIM).reshape(-1, 1)
potentials = {n: np.zeros(STATE_DIM) for n in names}
prior = {n: np.full(STATE_DIM, 1.0 / STATE_DIM) for n in names}
f_func = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.0)
jko = FullJKOSolver(
    f_functional=f_func,
    h=0.05,
    support=support,
    epsilon=0.05,
    max_iter=50,
    n_inner=5,
)
sim = ParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=SEED)

particles = sim.initialize()
out_dir = Path("bench/runs/T2_pairs")
out_dir.mkdir(parents=True, exist_ok=True)
for i in range(N_PAIRS):
    particles = sim.evolve(particles, dt=1e-3, n_steps=50, potential_fn=_zero)
    pre_state = _particles_to_flow_state(particles, support, names)
    post_state = jko.step(pre_state)
    pre = pre_state.rho["phono"]
    post = post_state.rho["phono"]
    save_file(
        {"rho::phono": pre.astype(np.float32), "rho::phono_next": post.astype(np.float32)},
        str(out_dir / f"pair_{i:04d}.safetensors"),
    )
print(f"Wrote {N_PAIRS} (pre, post) pairs under {out_dir}")
