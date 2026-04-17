"""Figure 4: KL divergence to JKO baseline as a function of Sinkhorn epsilon."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_kl_vs_epsilon(
    epsilons: list[float],
    kl_values: list[float],
    out_dir: Path,
    filename: str = "fig4_kl_vs_epsilon",
) -> Path:
    """Plot KL(T2 || baseline) against Sinkhorn epsilon on a log-log scale."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.loglog(epsilons, kl_values, marker="o")
    ax.set_xlabel("Sinkhorn epsilon")
    ax.set_ylabel("KL(T2 || baseline)")
    ax.set_title("Entropic Sinkhorn bias vs epsilon")
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path
