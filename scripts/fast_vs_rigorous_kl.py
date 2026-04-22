"""Quantitative comparison between fast path (plain JKO) and rigorous path
(FullJKOSolver with Wasserstein prox). Measures KL divergence between the
two final distributions on the same seeds to replace the paper's current
``qualitatively similar'' claim with a measured number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy

GRID = 16
N_PARTICLES = 500
N_FAST = 200
N_SLOW = 20
SEEDS = [0, 1, 2, 3, 4]


def _build_common_setup():
    species = CanonicalSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    potentials = {n: np.zeros(GRID) for n in names}
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f_func = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)
    return species, support, f_func


def run_fast(seed: int) -> dict[str, np.ndarray]:
    species, support, f_func = _build_common_setup()
    names = species.species_names()
    jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10, apply_w2_prox=False)
    sim = MLXParticleSimulator(species=species, n_particles=N_PARTICLES, latent_dim=2, seed=seed)
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=N_FAST, n_slow=N_SLOW, support=support)
    manifest = loop.run(seed=seed)
    final = manifest["trajectory"][-1]
    return {n: np.asarray(final.rho[n]) for n in names}


def run_rigorous(seed: int) -> dict[str, np.ndarray]:
    species, support, f_func = _build_common_setup()
    names = species.species_names()
    jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.05)
    sim = MLXParticleSimulator(species=species, n_particles=N_PARTICLES, latent_dim=2, seed=seed)
    loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=N_FAST, n_slow=N_SLOW, support=support)
    manifest = loop.run(seed=seed)
    final = manifest["trajectory"][-1]
    return {n: np.asarray(final.rho[n]) for n in names}


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_s = np.clip(p, 1e-12, None)
    q_s = np.clip(q, 1e-12, None)
    return float((p_s * np.log(p_s / q_s)).sum())


def main() -> None:
    per_seed: list[dict] = []
    for seed in SEEDS:
        fast = run_fast(seed)
        rigorous = run_rigorous(seed)
        kl_per_species = {n: kl_divergence(rigorous[n], fast[n]) for n in fast}
        total_kl = sum(kl_per_species.values())
        per_seed.append({"seed": seed, "kl_per_species": kl_per_species, "total_kl": total_kl})
        print(f"seed={seed}  total_KL(rigorous || fast)={total_kl:.6f}")

    kl_values = [entry["total_kl"] for entry in per_seed]
    summary = {
        "n_seeds": len(SEEDS),
        "params": {
            "grid": GRID,
            "n_particles": N_PARTICLES,
            "n_fast": N_FAST,
            "n_slow": N_SLOW,
        },
        "per_seed": per_seed,
        "mean_total_kl": float(np.mean(kl_values)),
        "std_total_kl": float(np.std(kl_values)),
        "max_total_kl": float(np.max(kl_values)),
    }
    Path("paper/fast_vs_rigorous_kl.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nMean KL(rigorous || fast) over {len(SEEDS)} seeds: "
        f"{summary['mean_total_kl']:.6f} +/- {summary['std_total_kl']:.6f} "
        f"(max {summary['max_total_kl']:.6f})"
    )
    print("Wrote paper/fast_vs_rigorous_kl.json")


if __name__ == "__main__":
    main()
