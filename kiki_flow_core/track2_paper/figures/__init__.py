"""Figure generators for the Track 2 paper run.

Applies a shared matplotlib rcParams tune on import so every figure
produced by the sub-modules uses the same font size, line widths, and
colour cycle.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.4,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    }
)

from kiki_flow_core.track2_paper.figures.continual_learning_gap import (  # noqa: E402
    make_continual_learning_gap,
)
from kiki_flow_core.track2_paper.figures.f_decay_curves import make_f_decay_curves  # noqa: E402
from kiki_flow_core.track2_paper.figures.fig7_cl_forgetting import (  # noqa: E402
    make_cl_forgetting,
)
from kiki_flow_core.track2_paper.figures.fig8_bridge_impact import (  # noqa: E402
    make_bridge_impact,
)
from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon  # noqa: E402
from kiki_flow_core.track2_paper.figures.phase_portrait import make_phase_portrait  # noqa: E402
from kiki_flow_core.track2_paper.figures.turing_patterns import make_turing_patterns  # noqa: E402

__all__ = [
    "make_continual_learning_gap",
    "make_f_decay_curves",
    "make_kl_vs_epsilon",
    "make_phase_portrait",
    "make_turing_patterns",
    "make_cl_forgetting",
    "make_bridge_impact",
]
