import numpy as np

from kiki_flow_core.species import OrthoSpecies
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator


def test_multiscale_loop_produces_manifest():
    species = OrthoSpecies()
    sim = ParticleSimulator(species=species, n_particles=200, latent_dim=2, seed=0)
    n = 8
    potentials = {name: np.zeros(n) for name in species.species_names()}
    prior = {name: np.full(n, 1.0 / n) for name in species.species_names()}
    f = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.0)
    support = np.linspace(-2, 2, n).reshape(-1, 1)
    jko = FullJKOSolver(
        f_functional=f, h=0.05, support=support, epsilon=0.05, max_iter=50, n_inner=5
    )
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=20, n_slow=3, support=support)
    manifest = loop.run(seed=0)
    assert manifest["n_slow_completed"] == 3  # noqa: PLR2004
    assert len(manifest["trajectory"]) == 3  # noqa: PLR2004
