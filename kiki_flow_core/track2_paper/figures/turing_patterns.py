"""Figure 3: Turing pattern emergence (species rho spatial profiles at final tau)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kiki_flow_core.state import FlowState  # noqa: E402


def make_turing_patterns(
    trajectory: list[FlowState],
    out_dir: Path,
    filename: str = "fig3_turing",
) -> Path:
    """Plot the final-tau spatial profiles for each species."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = trajectory[-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, rho in final.rho.items():
        xs = np.arange(rho.size)
        ax.plot(xs, rho, label=name)
    ax.set_xlabel("grid index")
    ax.set_ylabel("rho")
    ax.set_title("Final-tau species spatial profiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path
