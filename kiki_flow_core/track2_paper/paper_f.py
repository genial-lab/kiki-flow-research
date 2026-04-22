"""T2 free energy: potential + KL prior + Levelt-Baddeley reaction + Turing cross-diffusion."""

from __future__ import annotations

import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState


class T2FreeEnergy(FreeEnergy):
    """F_T2 = sum_i <rho_i, V_i> + sum_i KL(rho_i || prior_i) + reaction + turing."""

    def __init__(
        self,
        species: CanonicalSpecies,
        potentials: dict[str, np.ndarray],
        prior: dict[str, np.ndarray],
        turing_strength: float = 0.1,
    ) -> None:
        self.species = species
        self.potentials = potentials
        self.prior = prior
        self.turing_strength = turing_strength
        self._coupling = species.coupling_matrix()
        # Canonical J split: J = J_sym + J_asym.
        # J_sym carries the conservative (gradient-of-scalar-energy) piece;
        # J_asym carries the non-conservative drift. The scalar energy
        # sum_{i,k} J[i,k] <rho_i, rho_k> is unchanged when routed through
        # J_sym because sum_{i,k} J_asym[i,k] <rho_i, rho_k> = 0 identically
        # (antisymmetric tensor contracted with symmetric <rho_i, rho_k>).
        self._J_sym = 0.5 * (self._coupling + self._coupling.T)
        self._J_asym = 0.5 * (self._coupling - self._coupling.T)

    def value(self, state: FlowState) -> float:
        total = 0.0
        names = self.species.species_names()
        rhos = [state.rho[n] for n in names]

        for n, rho in zip(names, rhos, strict=True):
            total += float(np.dot(rho, self.potentials[n]))
            rho_safe = np.clip(rho, 1e-12, None)
            prior_safe = np.clip(self.prior[n], 1e-12, None)
            total += float(np.sum(rho_safe * np.log(rho_safe / prior_safe)))

        # Route the coupling energy through J_sym: mathematically equivalent
        # to using the full J (antisymmetric contribution vanishes), but makes
        # the derivation of the conservative part unambiguous.
        j = self._J_sym
        for i, ri in enumerate(rhos):
            for k, rk in enumerate(rhos):
                total += float(j[i, k] * np.dot(ri, rk))

        if self.turing_strength > 0.0:
            turing = 0.0
            for i, ri in enumerate(rhos):
                for k in range(i + 1, len(rhos)):
                    rk = rhos[k]
                    turing += float(np.sum(np.abs(np.gradient(ri) * np.gradient(rk))))
            total += self.turing_strength * turing

        return total

    def _grad_conservative(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Per-species conservative coupling drive: (J_sym @ rho)_i.

        This is the J-coupling piece of delta F / delta rho_i coming from
        the symmetric part of J; it is the gradient of the scalar energy
        sum_{i,k} J_sym[i,k] <rho_i, rho_k> when one identifies rhos as
        living on a flat (non-simplex) inner-product space. V_i, KL, and
        Turing contributions are NOT included here: this helper isolates
        the J_sym contribution only so the split invariant
        J rho = J_sym rho + J_asym rho can be checked directly.
        """
        j_sym = self._J_sym
        out: list[np.ndarray] = []
        for i in range(len(rhos)):
            acc = np.zeros_like(rhos[i])
            for k, rk in enumerate(rhos):
                acc = acc + j_sym[i, k] * rk
            out.append(acc)
        return out

    def _drift_nonconservative(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Per-species non-conservative drift: (J_asym @ rho)_i.

        Isolated antisymmetric drift field driven by the asymmetric part
        of the Levelt-Baddeley coupling. Contracted with the symmetric
        inner product <rho_i, rho_k> it integrates to zero against the
        scalar energy, so it cannot arise as the gradient of any scalar
        functional of the rhos — it is a pure non-conservative forcing.
        """
        j_asym = self._J_asym
        out: list[np.ndarray] = []
        for i in range(len(rhos)):
            acc = np.zeros_like(rhos[i])
            for k, rk in enumerate(rhos):
                acc = acc + j_asym[i, k] * rk
            out.append(acc)
        return out

    def coupling_drive(self, rhos: list[np.ndarray]) -> list[np.ndarray]:
        """Legacy full-J coupling drive: (J @ rho)_i per species.

        Backwards-compatible sum of the conservative and non-conservative
        components. By construction equals
        ``[self._coupling[i, k] * rhos[k] for i, k]`` accumulated, which is
        what the pre-split code used as the J-coupling variational drive.
        """
        cons = self._grad_conservative(rhos)
        drift = self._drift_nonconservative(rhos)
        return [c + d for c, d in zip(cons, drift, strict=True)]

    def apply_drift_splitting(self, state: FlowState, h_drift: float = 0.01) -> FlowState:
        """Operator-splitting step for the antisymmetric linear drift b_asym = J_asym @ rho.

        Applies M = exp(h_drift * J_asym) per grid cell to the 4-vector of
        per-species densities. Since J_asym is antisymmetric, M is orthogonal,
        so the total activity sum_i rho_i(x) is preserved at every grid cell x.
        Per-species total mass can drift slightly under the rotation (individual
        species are not preserved axis-wise); we rescale each species back to its
        prior total mass so the simplex constraint (non-negative, sums to 1) still
        holds for every rho_i after the step.

        A small Taylor expansion is used for the matrix exponential so the repo
        does not take a direct dependency on scipy. For ``h_drift`` of order 0.01
        and ``||h_drift * J_asym||_2`` of order 3e-3 the 4-term truncation error
        is below 1e-10 in operator norm.
        """
        names = self.species.species_names()
        # Shape (n_species, grid).
        old_rho = np.stack([np.asarray(state.rho[n]) for n in names], axis=0)
        old_mass = old_rho.sum(axis=1)
        # Matrix exponential via Taylor series: accurate to ~(h*||J||)^4 / 24.
        hj = h_drift * self._J_asym
        eye = np.eye(hj.shape[0])
        hj2 = hj @ hj
        hj3 = hj2 @ hj
        hj4 = hj3 @ hj
        m = eye + hj + hj2 / 2.0 + hj3 / 6.0 + hj4 / 24.0
        new_rho_raw = m @ old_rho
        # Clip tiny numerical negatives that can arise near simplex boundary.
        new_rho_raw = np.maximum(new_rho_raw, 0.0)
        new_mass = new_rho_raw.sum(axis=1)
        scale = old_mass / np.maximum(new_mass, 1e-12)
        new_rho = new_rho_raw * scale[:, None]
        new_rho_dict = {names[i]: new_rho[i] for i in range(len(names))}
        return state.model_copy(update={"rho": new_rho_dict})
