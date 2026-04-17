"""Figure 2: F decay curves across JKO steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kiki_flow_core.master_equation import FreeEnergy  # noqa: E402
from kiki_flow_core.state import FlowState  # noqa: E402


def make_f_decay_curves(
    trajectory: list[FlowState],
    f_functional: FreeEnergy,
    out_dir: Path,
    filename: str = "fig2_f_decay",
) -> Path:
    """Plot F[state] over tau for the given trajectory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = [s.tau for s in trajectory]
    values = [f_functional.value(s) for s in trajectory]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(taus, values, marker="o")
    ax.set_xlabel("tau")
    ax.set_ylabel("F")
    ax.set_title("T2 free-energy decay")
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path
