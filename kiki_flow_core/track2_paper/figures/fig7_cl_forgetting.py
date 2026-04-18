"""Fig 7: forgetting score per task, with/without bridge, on the full LLM bench."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_cl_forgetting(
    forgetting_without: dict[str, float],
    forgetting_with: dict[str, float],
    out_dir: Path,
    filename: str = "fig7_cl_forgetting",
) -> Path:
    """Generate grouped-bar chart of forgetting with and without bridge.

    Args:
        forgetting_without: task name -> forgetting score (no bridge).
        forgetting_with: task name -> forgetting score (with bridge).
        out_dir: output directory for PNG and PDF.
        filename: base filename (without extension).

    Returns:
        Path to PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(forgetting_without.keys())
    without = [forgetting_without[t] for t in tasks]
    with_ = [forgetting_with[t] for t in tasks]
    fig, ax = plt.subplots(figsize=(5, 3))
    width = 0.35
    xs = list(range(len(tasks)))
    ax.bar([x - width / 2 for x in xs], without, width=width, label="without bridge")
    ax.bar([x + width / 2 for x in xs], with_, width=width, label="with bridge")
    ax.set_xticks(xs)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("forgetting score (acc_before - acc_after)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / f"{filename}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png
