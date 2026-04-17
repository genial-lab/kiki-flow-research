"""Pure-NumPy neural surrogate forward pass; weights via safetensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def _gelu(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    return out


class NeuralSurrogate:
    """3-layer MLP with residual skip. Forward-only; training in surrogate_trainer.py."""

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        state_dim: int,
        embed_dim: int,
        hidden: int,
    ) -> None:
        self.weights = weights
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hidden = hidden

    @classmethod
    def load(cls, path: Path, state_dim: int, embed_dim: int, hidden: int) -> NeuralSurrogate:
        weights = load_file(str(path))
        return cls(weights, state_dim, embed_dim, hidden)

    def forward(self, state_flat: np.ndarray, query_embed: np.ndarray) -> np.ndarray:
        x = np.concatenate([state_flat, query_embed])
        w = self.weights
        h1 = _gelu(x @ w["w1"] + w["b1"])
        h2 = _gelu(h1 @ w["w2"] + w["b2"]) + h1
        h3 = h2 @ w["w3"] + w["b3"]
        delta: np.ndarray = np.tanh(h3).astype(np.float32)
        return delta
