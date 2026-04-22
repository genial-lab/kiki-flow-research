import numpy as np

from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator


def _zero_potential(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def test_particle_simulator_conserves_count():
    species = CanonicalSpecies()
    sim = ParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=0)
    particles = sim.initialize()
    assert particles["positions"].shape == (500, 2)  # noqa: PLR2004
    new_particles = sim.evolve(particles, dt=1e-3, n_steps=10, potential_fn=_zero_potential)
    assert new_particles["positions"].shape == particles["positions"].shape


def test_particle_simulator_deterministic_with_seed():
    species = CanonicalSpecies()
    sim1 = ParticleSimulator(species=species, n_particles=100, latent_dim=2, seed=42)
    sim2 = ParticleSimulator(species=species, n_particles=100, latent_dim=2, seed=42)
    p1 = sim1.initialize()
    p2 = sim2.initialize()
    np.testing.assert_allclose(p1["positions"], p2["positions"], atol=0.0)


def test_particle_simulator_species_tags_preserved():
    species = CanonicalSpecies()
    sim = ParticleSimulator(species=species, n_particles=200, latent_dim=2, seed=1)
    particles = sim.initialize()
    assert len(particles["species_tags"]) == 200  # noqa: PLR2004
    assert set(particles["species_tags"]).issubset({"phono", "lex", "syntax", "sem"})
