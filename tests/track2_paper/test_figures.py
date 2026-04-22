from pathlib import Path

import numpy as np

from kiki_flow_core.master_equation import ZeroF
from kiki_flow_core.species import CanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.figures import (
    make_continual_learning_gap,
    make_f_decay_curves,
    make_kl_vs_epsilon,
    make_phase_portrait,
    make_turing_patterns,
)


def fake_trajectory() -> list[FlowState]:
    return [
        FlowState(
            rho={n: np.array([0.5, 0.5]) for n in CanonicalSpecies().species_names()},
            P_theta=np.zeros(4),
            mu_curr=np.array([1.0]),
            tau=i,
            metadata={"track_id": "T2"},
        )
        for i in range(3)
    ]


def test_phase_portrait_writes_png(tmp_path: Path):
    out = make_phase_portrait(fake_trajectory(), out_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert (tmp_path / "fig1_phase_portrait.pdf").exists()


def test_f_decay_writes_png(tmp_path: Path):
    out = make_f_decay_curves(fake_trajectory(), f_functional=ZeroF(), out_dir=tmp_path)
    assert out.exists()


def test_turing_writes_png(tmp_path: Path):
    out = make_turing_patterns(fake_trajectory(), out_dir=tmp_path)
    assert out.exists()


def test_kl_vs_epsilon_writes_png(tmp_path: Path):
    out = make_kl_vs_epsilon(
        epsilons=[0.001, 0.01, 0.1],
        kl_values=[1.0, 0.3, 0.05],
        out_dir=tmp_path,
    )
    assert out.exists()


def test_cl_gap_writes_png(tmp_path: Path):
    out = make_continual_learning_gap(
        tasks=["task_a", "task_b"],
        with_consolidation=[0.85, 0.78],
        without_consolidation=[0.70, 0.60],
        out_dir=tmp_path,
    )
    assert out.exists()
