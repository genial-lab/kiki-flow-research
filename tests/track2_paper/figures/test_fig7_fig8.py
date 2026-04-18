"""Smoke tests for fig7 and fig8 generators."""

from __future__ import annotations

from pathlib import Path

from kiki_flow_core.track2_paper.figures.fig7_cl_forgetting import make_cl_forgetting
from kiki_flow_core.track2_paper.figures.fig8_bridge_impact import make_bridge_impact


def test_fig7_emits_png_and_pdf(tmp_path: Path) -> None:
    """Test fig7_cl_forgetting emits PNG and PDF outputs."""
    without = {"phono": 0.6, "lex": 0.4, "syn": 0.0}
    with_ = {"phono": 0.0, "lex": 0.5, "syn": 0.6}
    make_cl_forgetting(without, with_, out_dir=tmp_path)
    assert (tmp_path / "fig7_cl_forgetting.png").exists()
    assert (tmp_path / "fig7_cl_forgetting.pdf").exists()


def test_fig8_emits_png_and_pdf(tmp_path: Path) -> None:
    """Test fig8_bridge_impact emits PNG and PDF outputs."""
    trace = [
        {"step": 0, "advisory": [0.1, 0.2, 0.3]},
        {"step": 1, "advisory": None},
        {"step": 2, "advisory": [0.0, 0.1, 0.2]},
    ]
    make_bridge_impact(trace, out_dir=tmp_path)
    assert (tmp_path / "fig8_bridge_impact.png").exists()
    assert (tmp_path / "fig8_bridge_impact.pdf").exists()
