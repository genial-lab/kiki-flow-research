"""Dense hyperparameter sweep for issue #2 part 1.

Expands the original 27-config sweep in scripts/hyperparam_sweep.py to
a denser grid of 72 configurations. Produces paper/hyperparam_sweep_dense.json
and a fig6_heatmap.png for the camera-ready.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kiki_flow_core.master_equation import JKOStep  # noqa: E402
from kiki_flow_core.species import OrthoSpecies  # noqa: E402
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator  # noqa: E402
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: E402
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy  # noqa: E402

GRID = 16
MAX_ENTROPY_BITS = float(np.log2(GRID))
SEED = 0
ALPHAS = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0]
BETAS = [0.0, 0.5, 1.0, 2.0]
GAMMAS = [0.0, 1.0, 5.0]


def run_one(alpha: float, beta: float, gamma: float) -> dict[str, float]:
    species = OrthoSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    x = support[:, 0]
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


def plot_heatmap(results: list[dict], out_dir: Path) -> Path:
    """Produce fig6: heatmap of mean_entropy over (alpha, beta) at gamma=0."""
    results_gamma0 = [r for r in results if r["gamma"] == 0.0]
    alphas = sorted({r["alpha"] for r in results_gamma0})
    betas = sorted({r["beta"] for r in results_gamma0})
    grid = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            match = next(r for r in results_gamma0 if r["alpha"] == a and r["beta"] == b)
            grid[i, j] = match["mean_entropy_bits"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:.1f}" for b in betas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"$\beta$ (gradient descent strength)")
    ax.set_ylabel(r"$\alpha$ (potential attraction)")
    ax.set_title(r"Mean final entropy (bits) at $\gamma=0$, max=4")
    fig.colorbar(im, ax=ax, label="bits")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig6_heatmap.png"
    fig.savefig(png_path)
    fig.savefig(out_dir / "fig6_heatmap.pdf")
    plt.close(fig)
    return png_path


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
        if best is None or gap > best["gap_bits"]:
            best = entry

    assert best is not None
    print(
        f"Best configuration: alpha={best['alpha']} beta={best['beta']} "
        f"gamma={best['gamma']} mean_entropy={best['mean_entropy_bits']:.4f} "
        f"gap={best['gap_bits']:.4f} bits (out of max {MAX_ENTROPY_BITS:.4f})"
    )

    heatmap_path = plot_heatmap(results, Path("paper/figures"))
    print(f"Wrote heatmap: {heatmap_path}")

    Path("paper/hyperparam_sweep_dense.json").write_text(
        json.dumps(
            {
                "max_entropy_bits": MAX_ENTROPY_BITS,
                "n_configs": len(results),
                "grid": {"alpha": ALPHAS, "beta": BETAS, "gamma": GAMMAS},
                "results": results,
                "best": best,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
