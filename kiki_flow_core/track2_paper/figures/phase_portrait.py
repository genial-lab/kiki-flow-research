"""Figure 1: phase portrait of the 4 ortho species during a T2 run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kiki_flow_core.state import FlowState  # noqa: E402


def make_phase_portrait(
    trajectory: list[FlowState],
    out_dir: Path,
    filename: str = "fig1_phase_portrait",
) -> Path:
    """Render the 4-species phase portrait to PNG + PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["phono", "lex", "syntax", "sem"]
    centroids: list[tuple[float, float]] = []
    for state in trajectory:
        means = [float(state.rho[n].mean()) for n in names]
        centroids.append((means[0], means[1]))
    arr = np.asarray(centroids)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(arr[:, 0], arr[:, 1], marker="o", linewidth=0.8)
    ax.set_xlabel("mean rho_phono")
    ax.set_ylabel("mean rho_lex")
    ax.set_title("T2 phase portrait")
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    pdf_path = out_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path
