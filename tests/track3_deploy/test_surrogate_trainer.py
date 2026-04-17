from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.track3_deploy.surrogate_trainer import SurrogateTrainer


def _fake_trajectory(out_dir: Path, n_pairs: int = 20, state_dim: int = 32) -> None:
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_pairs):
        pre = rng.dirichlet(np.ones(state_dim)).astype(np.float32)
        post = pre + 0.01 * rng.standard_normal(state_dim).astype(np.float32)
        post = np.clip(post, 1e-6, None)
        post = post / post.sum()
        save_file(
            {"rho::phono": pre, "rho::phono_next": post.astype(np.float32)},
            str(out_dir / f"traj_{i:04d}.safetensors"),
        )


def test_trainer_writes_weights_after_training(tmp_path: Path):
    trajectory_dir = tmp_path / "traj"
    _fake_trajectory(trajectory_dir, n_pairs=30, state_dim=16)
    out = tmp_path / "weights.safetensors"
    trainer = SurrogateTrainer(
        mode="A",
        source_dir=trajectory_dir,
        state_dim=16,
        embed_dim=8,
        hidden=32,
        out_path=out,
    )
    metrics = trainer.train(epochs=3, lr=1e-3, batch_size=8)
    assert out.exists()
    assert "final_train_loss" in metrics
    assert np.isfinite(metrics["final_train_loss"])
