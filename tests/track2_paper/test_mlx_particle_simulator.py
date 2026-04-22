import mlx.core as mx
import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator


def test_mlx_particle_simulator_conserves_count():
    species = CanonicalSpecies()
    sim = MLXParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=0)
    particles = sim.initialize()
    assert particles["positions"].shape == (500, 2)  # noqa: PLR2004
    new_particles = sim.evolve(particles, dt=1e-3, n_steps=10, potential_fn=None)
    assert new_particles["positions"].shape == particles["positions"].shape


def test_mlx_particle_simulator_returns_mx_array():
    species = CanonicalSpecies()
    sim = MLXParticleSimulator(species=species, n_particles=100, latent_dim=2, seed=1)
    particles = sim.initialize()
    assert isinstance(particles["positions"], mx.array)


def test_mlx_particle_simulator_tags_in_species():
    species = CanonicalSpecies()
    sim = MLXParticleSimulator(species=species, n_particles=200, latent_dim=2, seed=2)
    particles = sim.initialize()
    assert set(particles["species_tags"]).issubset({"phono", "lex", "syntax", "sem"})


def test_mlx_particle_simulator_positions_change_after_evolve():
    species = CanonicalSpecies()
    sim = MLXParticleSimulator(species=species, n_particles=50, latent_dim=2, seed=3)
    p0 = sim.initialize()
    p1 = sim.evolve(p0, dt=1e-2, n_steps=20, potential_fn=None)
    p0_np = np.asarray(p0["positions"])
    p1_np = np.asarray(p1["positions"])
    assert not np.allclose(p0_np, p1_np, atol=1e-6)
