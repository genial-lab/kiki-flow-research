"""Phase-A pre-training: fit g_JEPA to (heuristic-label, encoder-embedding) pairs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import load_file, save_file

INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 384
_WEIGHT_INIT_SCALE = 0.02
_DEFAULT_LR = 3e-4
_DEFAULT_BATCH = 64
_DEFAULT_EPOCHS = 5

logger = logging.getLogger(__name__)


def gjepa_init_params(seed: int = 0) -> dict[str, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)
    return {
        "W1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * _WEIGHT_INIT_SCALE,
        "b1": jnp.zeros(HIDDEN_DIM),
        "W2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * _WEIGHT_INIT_SCALE,
        "b2": jnp.zeros(OUTPUT_DIM),
    }


def gjepa_forward(params: dict[str, jnp.ndarray], rho_flat: jnp.ndarray) -> jnp.ndarray:
    h = jax.nn.gelu(rho_flat @ params["W1"] + params["b1"])
    out: jnp.ndarray = h @ params["W2"] + params["b2"]
    return out


def _loss_fn(
    params: dict[str, jnp.ndarray],
    rho_flat: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    pred = gjepa_forward(params, rho_flat)
    loss: jnp.ndarray = jnp.mean((pred - targets) ** 2)
    return loss


def gjepa_step(
    params: dict[str, jnp.ndarray],
    opt_state: optax.OptState,
    rho_flat: np.ndarray,
    targets: np.ndarray,
    optim: optax.GradientTransformation,
) -> tuple[dict[str, jnp.ndarray], optax.OptState, float]:
    """One training step. Returns updated params, optimizer state, and pre-step loss."""
    rho_j = jnp.asarray(rho_flat)
    tgt_j = jnp.asarray(targets)
    loss_val, grads = jax.value_and_grad(_loss_fn)(params, rho_j, tgt_j)
    updates, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, float(loss_val)


def save_gjepa(params: dict[str, Any], path: Path | str) -> None:
    flat = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    save_file(flat, str(path))


def load_gjepa(path: Path | str) -> dict[str, jnp.ndarray]:
    flat = load_file(str(path))
    return {k: jnp.asarray(v) for k, v in flat.items()}


def train(
    labels_npz: Path,
    embeddings_npz: Path,
    output_path: Path,
    lr: float = _DEFAULT_LR,
    batch: int = _DEFAULT_BATCH,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 0,
) -> None:
    """Train g_JEPA on paired (heuristic-label, encoder-embedding) data."""
    labels = np.load(labels_npz, allow_pickle=True)
    embeddings = np.load(embeddings_npz, allow_pickle=True)
    hashes = sorted(set(labels.files) & set(embeddings.files))
    rho_flat = np.stack([labels[h].flatten() for h in hashes]).astype(np.float32)
    targets = np.stack([embeddings[h] for h in hashes]).astype(np.float32)
    logger.info("loaded %d paired samples", len(hashes))

    params = gjepa_init_params(seed=seed)
    optim = optax.adamw(lr)
    opt_state = optim.init(params)
    rng = np.random.default_rng(seed)
    n = rho_flat.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            params, opt_state, loss = gjepa_step(
                params, opt_state, rho_flat[idx], targets[idx], optim
            )
            total_loss += loss
            n_batches += 1
        logger.info("epoch=%d avg_loss=%.5f", epoch, total_loss / max(1, n_batches))
    save_gjepa(params, output_path)
    logger.info("saved g_JEPA to %s", output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-train g_JEPA on heuristic labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    train(
        args.labels,
        args.embeddings,
        args.output,
        lr=args.lr,
        batch=args.batch,
        epochs=args.epochs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
