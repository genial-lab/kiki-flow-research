from pathlib import Path

from kiki_flow_core.track2_paper.paper_run import run_paper


def test_paper_run_smoke(tmp_path: Path):
    stats = run_paper(
        seeds=[0, 1],
        n_particles=100,
        n_fast=10,
        n_slow=3,
        grid_size=16,
        out_dir=tmp_path,
    )
    assert (tmp_path / "stats.json").exists()
    assert "n_seeds" in stats
    assert stats["n_seeds"] == 2  # noqa: PLR2004
