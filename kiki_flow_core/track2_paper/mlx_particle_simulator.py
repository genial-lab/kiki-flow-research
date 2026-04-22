"""MLX-backed N-particle Langevin simulator for Track 2 (Apple Silicon GPU)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import mlx.core as mx
import numpy as np

from kiki_flow_core.species import CanonicalSpecies


class MLXParticleBatch(TypedDict):
    positions: mx.array
    species_tags: list[str]


class MLXParticleSimulator:
    """Langevin dynamics on the Metal GPU via MLX.

    Interface mirrors the numpy ``ParticleSimulator`` so it can be substituted
    in ``MultiscaleLoop`` without further plumbing, provided the downstream
    ``_particles_to_flow_state`` helper converts positions to numpy (which
    this revision of multiscale_loop does via ``np.asarray``).
    """

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
        self.seed = seed
        mx.random.seed(seed)
        self._np_rng = np.random.default_rng(seed)

    def initialize(self) -> MLXParticleBatch:
        positions = mx.random.normal(shape=(self.n, self.d))
        mx.eval(positions)
        names = self.species.species_names()
        tags = [names[int(self._np_rng.integers(0, len(names)))] for _ in range(self.n)]
        return {"positions": positions, "species_tags": tags}

    def evolve(
        self,
        particles: MLXParticleBatch,
        dt: float,
        n_steps: int,
        potential_fn: Callable[[mx.array], mx.array] | None = None,
    ) -> MLXParticleBatch:
        positions = particles["positions"]
        noise_coef = float(np.sqrt(dt)) * self.noise_scale
        for _ in range(n_steps):
            if potential_fn is not None:
                drift = -potential_fn(positions)
                positions = positions + dt * drift
            noise = mx.random.normal(shape=positions.shape) * noise_coef
            positions = positions + noise
        mx.eval(positions)
        return {"positions": positions, "species_tags": particles["species_tags"]}
