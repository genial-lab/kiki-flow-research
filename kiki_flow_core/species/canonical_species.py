"""Canonical 4-species decomposition (phono, lex, syntax, sem) with fixed Levelt-Baddeley J."""

from __future__ import annotations

from importlib import resources

import numpy as np
import yaml

from kiki_flow_core.species.base import SpeciesBase

_COUPLING_VARIANTS: dict[str, str] = {
    "dell": "dell_baddeley_coupling.yaml",
    "levelt": "levelt_baddeley_coupling.yaml",
}


class CanonicalSpecies(SpeciesBase):
    """Canonical four-species Levelt-Baddeley decomposition (phono, lex, syntax, sem).

    The densities live on a shared spatial grid; no L2 orthogonality is
    imposed. The term "canonical" is functional, referring to these being the
    four production-stage species of the Levelt 1999 architecture, not
    geometric.

    Literature-derived coupling is loaded from
    ``kiki_flow_core/species/data/``.

    Parameters
    ----------
    coupling_variant:
        Which published architecture to instantiate. ``"dell"`` (default)
        loads ``dell_baddeley_coupling.yaml`` and corresponds to the
        interactive spreading-activation network of Dell (1986), with a
        strong direct sem->lex path. ``"levelt"`` loads
        ``levelt_baddeley_coupling.yaml`` and corresponds to the serial
        WEAVER++ architecture of Levelt, Roelofs & Meyer (1999), with a
        strong syntactic detour (sem->syntax->lex) instead.

        Default is ``"dell"`` so all existing callers keep the same matrix.
    """

    def __init__(self, coupling_variant: str = "dell") -> None:
        if coupling_variant not in _COUPLING_VARIANTS:
            valid = ", ".join(sorted(_COUPLING_VARIANTS))
            raise ValueError(
                f"Unknown coupling_variant {coupling_variant!r}; expected one of: {valid}"
            )
        filename = _COUPLING_VARIANTS[coupling_variant]
        data_pkg = resources.files("kiki_flow_core.species.data")
        yaml_text = data_pkg.joinpath(filename).read_text()
        cfg = yaml.safe_load(yaml_text)
        self._variant: str = coupling_variant
        self._names: list[str] = list(cfg["species_order"])
        self._j = np.asarray(cfg["coupling_matrix"], dtype=np.float64)
        if self._j.shape != (4, 4):
            raise ValueError(f"Expected 4x4 coupling matrix, got {self._j.shape}")

    @property
    def coupling_variant(self) -> str:
        return self._variant

    def species_names(self) -> list[str]:
        return list(self._names)

    def coupling_matrix(self) -> np.ndarray:
        return self._j.copy()
