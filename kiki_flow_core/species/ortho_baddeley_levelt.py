"""4-species orthophonic decomposition (phono, lex, syntax, sem) with fixed Levelt-Baddeley J."""

from __future__ import annotations

from importlib import resources

import numpy as np
import yaml

from kiki_flow_core.species.base import SpeciesBase


class OrthoSpecies(SpeciesBase):
    """Levelt-Baddeley language production model.

    Four species (phono, lex, syntax, sem) with literature-derived coupling
    loaded from ``kiki_flow_core/species/data/levelt_baddeley_coupling.yaml``.
    """

    def __init__(self) -> None:
        data_pkg = resources.files("kiki_flow_core.species.data")
        yaml_text = data_pkg.joinpath("levelt_baddeley_coupling.yaml").read_text()
        cfg = yaml.safe_load(yaml_text)
        self._names: list[str] = list(cfg["species_order"])
        self._j = np.asarray(cfg["coupling_matrix"], dtype=np.float64)
        if self._j.shape != (4, 4):
            raise ValueError(f"Expected 4x4 coupling matrix, got {self._j.shape}")

    def species_names(self) -> list[str]:
        return list(self._names)

    def coupling_matrix(self) -> np.ndarray:
        return self._j.copy()
