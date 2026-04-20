"""TinyTransformer encoder: 4-layer, 8-head transformer over byte-level tokens.

JAX + flax implementation. ~8M params. Tokenization is byte-level (258 symbols:
256 bytes + BOS + PAD), avoiding external tokenizer deps. Max sequence length 128.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors.numpy import load_file, save_file

from kiki_flow_core.track3_deploy.encoders import register
from kiki_flow_core.track3_deploy.encoders.base import TextEncoder

VOCAB_SIZE = 258  # 256 bytes + BOS + PAD
BOS_TOKEN = 256
PAD_TOKEN = 257
MAX_SEQ_LEN = 128

_DEFAULT_D_MODEL = 256
_DEFAULT_N_LAYERS = 4
_DEFAULT_N_HEADS = 8
_DEFAULT_D_FF = 1024
_DEFAULT_OUTPUT_DIM = 384
_POS_EMB_INIT_SCALE = 0.02
_POOLING_EPS = 1e-8


def _tokenize(text: str) -> np.ndarray:
    """Byte-level tokenization with BOS prefix and PAD fill to MAX_SEQ_LEN."""
    b = text.encode("utf-8")[: MAX_SEQ_LEN - 1]
    arr = np.full(MAX_SEQ_LEN, PAD_TOKEN, dtype=np.int32)
    arr[0] = BOS_TOKEN
    if b:
        arr[1 : 1 + len(b)] = np.frombuffer(b, dtype=np.uint8)
    return arr


class _TinyTFBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # Pre-norm attention
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            deterministic=True,
        )(h, mask=mask)
        x = x + h
        # Pre-norm FFN
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


class _TinyTFModule(nn.Module):
    d_model: int = _DEFAULT_D_MODEL
    n_layers: int = _DEFAULT_N_LAYERS
    n_heads: int = _DEFAULT_N_HEADS
    d_ff: int = _DEFAULT_D_FF
    output_dim: int = _DEFAULT_OUTPUT_DIM

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        emb = nn.Embed(VOCAB_SIZE, self.d_model)(tokens)
        pos = self.param(
            "pos",
            lambda k: jax.random.normal(k, (MAX_SEQ_LEN, self.d_model)) * _POS_EMB_INIT_SCALE,
        )
        x = emb + pos[None, : tokens.shape[1], :]
        mask = (tokens != PAD_TOKEN)[:, None, None, :]
        for _ in range(self.n_layers):
            x = _TinyTFBlock(d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff)(x, mask)
        x = nn.LayerNorm()(x)
        # Mean-pool over non-pad positions
        mask_f = (tokens != PAD_TOKEN).astype(jnp.float32)[:, :, None]
        pooled = (x * mask_f).sum(axis=1) / (mask_f.sum(axis=1) + _POOLING_EPS)
        return nn.Dense(self.output_dim)(pooled)


def _flatten_params(prefix: str, tree: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            sub_prefix = f"{prefix}/{k}" if prefix else k
            out.update(_flatten_params(sub_prefix, v))
    else:
        out[prefix] = np.asarray(tree)
    return out


def _unflatten_params(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split("/")
        d = tree
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = jnp.asarray(v)
    return tree


@register("D_tiny_tf")
class EncoderD_TinyTransformer(TextEncoder):  # noqa: N801
    """Flax tiny-transformer wrapper with NumPy-friendly API."""

    def __init__(self, seed: int = 0) -> None:
        self.module = _TinyTFModule()
        key = jax.random.PRNGKey(seed)
        dummy = jnp.full((1, MAX_SEQ_LEN), PAD_TOKEN, dtype=jnp.int32).at[:, 0].set(BOS_TOKEN)
        self.params = self.module.init(key, dummy)
        self._apply_jit = jax.jit(self.module.apply)

    def encode(self, texts: list[str]) -> np.ndarray:
        tokens = np.stack([_tokenize(t) for t in texts])
        out = self._apply_jit(self.params, jnp.asarray(tokens))
        return np.asarray(out, dtype=np.float32)

    def param_count(self) -> int:
        return int(sum(p.size for p in jax.tree_util.tree_leaves(self.params)))

    def save(self, path: Path | str) -> None:
        flat = _flatten_params("", self.params)
        save_file(flat, str(path))

    def load(self, path: Path | str) -> None:
        flat = load_file(str(path))
        self.params = _unflatten_params(flat)
        self._apply_jit = jax.jit(self.module.apply)
