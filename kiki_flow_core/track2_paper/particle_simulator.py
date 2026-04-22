"""N-particle Langevin dynamics simulator for Track 2 paper track."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np

from kiki_flow_core.species import CanonicalSpecies


class ParticleBatch(TypedDict):
    positions: np.ndarray
    species_tags: list[str]


class ParticleSimulator:
    """Advance a batch of N particles tagged by species under Langevin dynamics."""

    def __init__(
        self,
        species: CanonicalSpecies,
        n_particles: int,
        latent_dim: int = 2,
        seed: int = 0,
        noise_scale: float = 0.05,
    ) -> None:
        self.species = species
        self.n = n_particles
        self.d = latent_dim
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def initialize(self) -> ParticleBatch:
        positions = self._rng.standard_normal((self.n, self.d))
        names = self.species.species_names()
        tags = [names[int(self._rng.integers(0, len(names)))] for _ in range(self.n)]
        return {"positions": positions, "species_tags": tags}

    def evolve(
        self,
        particles: ParticleBatch,
        dt: float,
        n_steps: int,
        potential_fn: Callable[[np.ndarray], np.ndarray],
    ) -> ParticleBatch:
        positions = particles["positions"].copy()
        for _ in range(n_steps):
            drift = -potential_fn(positions)
            noise = self._rng.standard_normal(positions.shape) * np.sqrt(dt) * self.noise_scale
            positions = positions + dt * drift + noise
        return {"positions": positions, "species_tags": particles["species_tags"]}
