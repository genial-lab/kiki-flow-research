"""Train a neural surrogate approximating a track's JKO flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

Mode = Literal["A", "B"]


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def _forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    h1 = _gelu(x @ params["w1"] + params["b1"])
    h2 = _gelu(h1 @ params["w2"] + params["b2"]) + h1
    h3 = h2 @ params["w3"] + params["b3"]
    return jnp.tanh(h3)


class SurrogateTrainer:
    """Distill a target track's JKO flow into a compact MLP."""

    def __init__(
        self,
        mode: Mode,
        source_dir: Path,
        state_dim: int,
        embed_dim: int,
        hidden: int,
        out_path: Path,
        seed: int = 0,
    ) -> None:
        self.mode = mode
        self.source_dir = Path(source_dir)
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.out_path = Path(out_path)
        self.seed = seed

    def _load_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        pre_list: list[np.ndarray] = []
        post_list: list[np.ndarray] = []
        for ckpt in sorted(self.source_dir.glob("*.safetensors")):
            tensors = load_file(str(ckpt))
            pre = tensors.get("rho::phono")
            post = tensors.get("rho::phono_next")
            if pre is not None and post is not None:
                pre_list.append(pre)
                post_list.append(post)
        if not pre_list:
            raise RuntimeError(f"No training pairs found in {self.source_dir}")
        return np.stack(pre_list), np.stack(post_list)

    def train(self, epochs: int, lr: float, batch_size: int) -> dict[str, Any]:
        pre, post = self._load_pairs()
        rng = np.random.default_rng(self.seed)
        in_dim = self.state_dim + self.embed_dim
        scale = 0.01
        params = {
            "w1": (rng.standard_normal((in_dim, self.hidden)) * scale).astype(np.float32),
            "b1": np.zeros(self.hidden, dtype=np.float32),
            "w2": (rng.standard_normal((self.hidden, self.hidden)) * scale).astype(np.float32),
            "b2": np.zeros(self.hidden, dtype=np.float32),
            "w3": (rng.standard_normal((self.hidden, self.state_dim)) * scale).astype(np.float32),
            "b3": np.zeros(self.state_dim, dtype=np.float32),
        }
        params_j = {k: jnp.asarray(v) for k, v in params.items()}
        q_embed = jnp.zeros(self.embed_dim, dtype=jnp.float32)

        def loss_fn(
            params: dict[str, jnp.ndarray], batch_pre: jnp.ndarray, batch_post: jnp.ndarray
        ) -> jnp.ndarray:
            x = jnp.concatenate(
                [batch_pre, jnp.broadcast_to(q_embed, (batch_pre.shape[0], self.embed_dim))],
                axis=1,
            )
            pred_delta = _forward(params, x)
            true_delta = batch_post - batch_pre
            return jnp.mean((pred_delta - true_delta) ** 2)

        grad_fn = jax.grad(loss_fn)
        n = pre.shape[0]
        final_loss = float("nan")
        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                bp = jnp.asarray(pre[idx])
                bpo = jnp.asarray(post[idx])
                grads = grad_fn(params_j, bp, bpo)
                params_j = {k: v - lr * grads[k] for k, v in params_j.items()}
            final_loss = float(loss_fn(params_j, jnp.asarray(pre), jnp.asarray(post)))

        out_tensors = {k: np.asarray(v) for k, v in params_j.items()}
        save_file(out_tensors, str(self.out_path))
        return {"final_train_loss": final_loss, "n_pairs": n, "mode": self.mode}
