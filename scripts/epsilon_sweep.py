"""Sweep Sinkhorn epsilon and measure the entropic bias on the prox step.

Uses a peaked initial distribution (delta at the left vs the right half of the
grid) and measures the bias of entropic-Sinkhorn prox vs a near-exact OT
reference (epsilon -> 0). Produces paper/figures/fig4_kl_vs_epsilon.{png,pdf}
with real data replacing the earlier synthetic placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon
from kiki_flow_core.wasserstein_ops import prox_w2

EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]
GRID = 16
SEED = 0


def run_one(eps: float) -> np.ndarray:
    """Apply prox_w2 to a smoothly peaked distribution, return result."""
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    x = support[:, 0]
    # Smooth Gaussian peak on the left
    distribution = np.exp(-0.5 * ((x + 1.0) / 0.5) ** 2)
    distribution = distribution / distribution.sum()
    # Smooth Gaussian peak on the right
    reference = np.exp(-0.5 * ((x - 1.0) / 0.5) ** 2)
    reference = reference / reference.sum()
    return prox_w2(distribution, reference=reference, epsilon=eps, support=support, n_iter=50)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_s = np.clip(p, 1e-12, None)
    q_s = np.clip(q, 1e-12, None)
    return float((p_s * np.log(p_s / q_s)).sum())


def main() -> None:
    baseline = run_one(EPSILONS[0])  # smallest epsilon = closest to exact OT
    kl_values: list[float] = []
    for eps in EPSILONS:
        rho = run_one(eps)
        kl_val = kl_divergence(rho, baseline)
        kl_values.append(max(kl_val, 1e-12))  # positive for log-scale plot
        print(f"eps={eps:.4f}  KL(eps || baseline)={kl_val:.6e}")

    out_dir = Path("paper/figures")
    make_kl_vs_epsilon(EPSILONS, kl_values, out_dir=out_dir, filename="fig4_kl_vs_epsilon")

    manifest_path = Path("paper/epsilon_sweep.json")
    manifest_path.write_text(
        json.dumps(
            {
                "epsilons": EPSILONS,
                "kl_values": kl_values,
                "params": {
                    "grid": GRID,
                    "seed": SEED,
                    "setup": "peaked-left vs peaked-right 1D prox",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {manifest_path} and updated fig4")


if __name__ == "__main__":
    main()
