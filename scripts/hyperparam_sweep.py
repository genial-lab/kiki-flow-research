"""Hyperparameter sweep over (alpha, beta, gamma) to find a regime that
induces specialization (entropy below max) in the T2 flow.

Runs a reduced problem (grid 16, n_slow 30, 500 particles) across a small
grid of coefficient values and reports the final entropy per species.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
MAX_ENTROPY_BITS = float(np.log2(GRID))
SEED = 0
ALPHAS = [0.0, 1.0, 5.0]
BETAS = [0.0, 0.1, 1.0]
GAMMAS = [0.0, 1.0, 5.0]


def run_one(alpha: float, beta: float, gamma: float) -> dict[str, float]:
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    x = support[:, 0]
    # Asymmetric potentials: each species prefers a different spatial region
    potentials = {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f_func = T2FreeEnergy(
        species=species, potentials=potentials, prior=prior, turing_strength=gamma * 0.1
    )
    # Inject beta into gradient-descent strength by scaling n_inner
    n_inner = max(1, int(10 * beta)) if beta > 0 else 1
    jko = JKOStep(
        f_functional=f_func, h=0.05, support=support, n_inner=n_inner, apply_w2_prox=False
    )
    sim = MLXParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=SEED)
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=200, n_slow=30, support=support)
    manifest = loop.run(seed=SEED)
    final = manifest["trajectory"][-1]
    entropies = {}
    for n, rho in final.rho.items():
        rho_s = np.clip(np.asarray(rho), 1e-12, None)
        entropies[n] = float(-(rho_s * np.log2(rho_s)).sum())
    return entropies


def main() -> None:
    results: list[dict] = []
    best: dict | None = None
    for alpha, beta, gamma in itertools.product(ALPHAS, BETAS, GAMMAS):
        entropies = run_one(alpha, beta, gamma)
        mean_entropy = float(np.mean(list(entropies.values())))
        gap = MAX_ENTROPY_BITS - mean_entropy
        entry = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "mean_entropy_bits": mean_entropy,
            "gap_bits": gap,
            "per_species": entropies,
        }
        results.append(entry)
        print(
            f"alpha={alpha:.1f} beta={beta:.2f} gamma={gamma:.1f}  "
            f"H_mean={mean_entropy:.4f}  gap={gap:.4f}"
        )
        if best is None or gap > best["gap_bits"]:
            best = entry

    Path("paper/hyperparam_sweep.json").write_text(
        json.dumps(
            {
                "max_entropy_bits": MAX_ENTROPY_BITS,
                "results": results,
                "best": best,
            },
            indent=2,
        )
    )
    assert best is not None
    print(
        f"\nBest configuration: alpha={best['alpha']} beta={best['beta']} "
        f"gamma={best['gamma']} mean_entropy={best['mean_entropy_bits']:.4f} "
        f"gap={best['gap_bits']:.4f} bits"
    )


if __name__ == "__main__":
    main()
