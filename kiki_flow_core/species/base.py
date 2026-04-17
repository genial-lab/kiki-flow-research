"""Abstract base for species definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SpeciesBase(ABC):
    """Abstract species set with a coupling structure used by reaction-diffusion."""

    @abstractmethod
    def species_names(self) -> list[str]:
        """Ordered list of species names."""

    @abstractmethod
    def coupling_matrix(self) -> np.ndarray:
        """2D coupling matrix of shape (n_species, n_species).

        J[i, j] = strength from sender j to receiver i.
        """

    @property
    def n_species(self) -> int:
        return len(self.species_names())
