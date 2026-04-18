"""Fig 8: time-series of advisory magnitude correlated with forgetting events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_bridge_impact(
    advisory_trace: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "fig8_bridge_impact",
) -> Path:
    """Generate time-series plot of advisory L1 magnitude over training steps.

    Args:
        advisory_trace: list of dicts with "step" (int) and "advisory" (list or None).
        out_dir: output directory for PNG and PDF.
        filename: base filename (without extension).

    Returns:
        Path to PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in advisory_trace]
    magnitudes = [
        sum(abs(v) for v in entry["advisory"]) if entry["advisory"] else 0.0
        for entry in advisory_trace
    ]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(steps, magnitudes, linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("advisory L1 magnitude")
    ax.set_title("Bridge advisory signal over training")
    fig.tight_layout()
    png = out_dir / f"{filename}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png
