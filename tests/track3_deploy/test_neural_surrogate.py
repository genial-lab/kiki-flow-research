from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate


def _write_dummy_weights(
    path: Path, state_dim: int = 32, embed_dim: int = 384, hidden: int = 64
) -> None:
    rng = np.random.default_rng(0)
    tensors = {
        "w1": (rng.standard_normal((state_dim + embed_dim, hidden)) * 0.01).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.standard_normal((hidden, hidden)) * 0.01).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "w3": (rng.standard_normal((hidden, state_dim)) * 0.01).astype(np.float32),
        "b3": np.zeros(state_dim, dtype=np.float32),
    }
    save_file(tensors, str(path))


def test_surrogate_forward_bounded(tmp_path: Path):
    path = tmp_path / "weights.safetensors"
    _write_dummy_weights(path, state_dim=32, hidden=64)
    surr = NeuralSurrogate.load(path, state_dim=32, embed_dim=384, hidden=64)
    state_flat = np.full(32, 1.0 / 32, dtype=np.float32)
    query = np.zeros(384, dtype=np.float32)
    delta = surr.forward(state_flat, query)
    assert delta.shape == (32,)  # noqa: PLR2004
    assert np.isfinite(delta).all()
    assert np.abs(delta).max() <= 1.0 + 1e-5  # tanh bounded  # noqa: PLR2004
