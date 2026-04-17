"""Figure 5: continual-learning gap vs a no-consolidation baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_continual_learning_gap(
    tasks: list[str],
    with_consolidation: list[float],
    without_consolidation: list[float],
    out_dir: Path,
    filename: str = "fig5_cl_gap",
) -> Path:
    """Bar chart comparing continual-learning performance with vs without T2 consolidation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    width = 0.35
    xs = range(len(tasks))
    ax.bar([x - width / 2 for x in xs], with_consolidation, width=width, label="with T2")
    ax.bar([x + width / 2 for x in xs], without_consolidation, width=width, label="without")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Continual learning: T2 vs baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path
