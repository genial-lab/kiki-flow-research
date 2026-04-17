"""Figure generators for the Track 2 paper run."""

from kiki_flow_core.track2_paper.figures.continual_learning_gap import (
    make_continual_learning_gap,
)
from kiki_flow_core.track2_paper.figures.f_decay_curves import make_f_decay_curves
from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon
from kiki_flow_core.track2_paper.figures.phase_portrait import make_phase_portrait
from kiki_flow_core.track2_paper.figures.turing_patterns import make_turing_patterns

__all__ = [
    "make_continual_learning_gap",
    "make_f_decay_curves",
    "make_kl_vs_epsilon",
    "make_phase_portrait",
    "make_turing_patterns",
]
