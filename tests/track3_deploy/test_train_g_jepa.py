"""Tests for g_JEPA phase-A pre-training."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from kiki_flow_core.track3_deploy.train_g_jepa import (
    gjepa_init_params,
    gjepa_step,
    load_gjepa,
    save_gjepa,
)

HIDDEN_DIM = 256
INPUT_DIM = 128
OUTPUT_DIM = 384
BATCH = 8
LR = 1e-2
N_STEPS = 50
LOSS_REDUCTION = 0.5


def test_init_shapes() -> None:
    params = gjepa_init_params(seed=0)
    assert params["W1"].shape == (INPUT_DIM, HIDDEN_DIM)
    assert params["b1"].shape == (HIDDEN_DIM,)
    assert params["W2"].shape == (HIDDEN_DIM, OUTPUT_DIM)
    assert params["b2"].shape == (OUTPUT_DIM,)


def test_steps_reduce_loss() -> None:
    """N steps of optax AdamW on a fixed toy batch must reduce MSE loss > 50%."""
    import optax  # noqa: PLC0415

    params = gjepa_init_params(seed=0)
    rng = np.random.default_rng(0)
    rho_flat = rng.random((BATCH, INPUT_DIM)).astype(np.float32)
    rho_flat /= rho_flat.sum(axis=-1, keepdims=True)
    targets = rng.standard_normal((BATCH, OUTPUT_DIM)).astype(np.float32)
    optim = optax.adamw(LR)
    opt_state = optim.init(params)
    loss_before = None
    loss_after = None
    for i in range(N_STEPS):
        params, opt_state, loss = gjepa_step(params, opt_state, rho_flat, targets, optim)
        if i == 0:
            loss_before = loss
        loss_after = loss
    assert loss_before is not None and loss_after is not None
    assert loss_after < loss_before * LOSS_REDUCTION, f"{loss_before} -> {loss_after}"


def test_save_load_roundtrip(tmp_path) -> None:
    params = gjepa_init_params(seed=0)
    path = tmp_path / "g_jepa.safetensors"
    save_gjepa(params, path)
    loaded = load_gjepa(path)
    for key in params:
        np.testing.assert_array_equal(np.asarray(params[key]), np.asarray(loaded[key]))
