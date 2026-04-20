"""In-repo reference implementation of the KikiFlowBridge.

This mirrors the design in docs/superpowers/patches/02-micro-kiki-dep3
-runner-factory.md, but shipped inside kiki-flow-research so it can be
unit-tested against a mock router. When micro-kiki integrates, they can
either import this class as a dependency or copy it into their serving
layer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from kiki_flow_core.hooks import RoutingAdapter
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
from kiki_flow_core.track3_deploy.streaming_runner import StreamingRunner

logger = logging.getLogger(__name__)

N_STACKS_DEFAULT = 32
N_ORTHO = 4


class KikiFlowBridge:
    """Lazy-loaded StreamingRunner wrapper with its own MiniLM tokenizer.

    Returns ``None`` from ``route_advisory()`` if disabled, NaN, dim
    mismatch, weights missing, or any other failure; callers must treat
    the result as purely advisory.

    Flag: ``KIKI_FLOW_ENABLED=1`` in the environment enables the bridge;
    default is disabled.
    """

    def __init__(
        self,
        weights_path: Path,
        state_dim: int = 128,
        embed_dim: int = 384,
        hidden: int = 256,
        n_stacks: int = N_STACKS_DEFAULT,
        use_stub_encoder: bool = True,
    ) -> None:
        self.enabled = os.environ.get("KIKI_FLOW_ENABLED", "0") == "1"
        self.n_stacks = n_stacks
        self._runner: StreamingRunner | None = None
        if not self.enabled:
            return
        try:
            surrogate = NeuralSurrogate.load(
                weights_path, state_dim=state_dim, embed_dim=embed_dim, hidden=hidden
            )
            encoder = QueryEncoder(use_stub=use_stub_encoder)
            species = ["phono:code", "sem:code", "lex:code", "syntax:code"]
            initial = FlowState(
                rho={sp: np.ones(n_stacks, dtype=np.float32) / n_stacks for sp in species},
                P_theta=np.zeros(8),
                mu_curr=np.array([1.0]),
                tau=0,
                metadata={"track_id": "T3"},
            )
            self._runner = StreamingRunner(
                surrogate=surrogate,
                encoder=encoder,
                routing_adapter=RoutingAdapter(publisher=lambda _adv: None),
                initial_state=initial,
            )
            logger.info("kiki-flow bridge ready, state_dim=%d", state_dim)
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow bridge init failed, disabling: %s", e)
            self.enabled = False
            self._runner = None

    def route_advisory(self, query: str) -> np.ndarray | None:
        """Return a ``(n_stacks,)``-shaped array of stack advisory weights."""
        if not self.enabled or self._runner is None:
            return None
        try:
            advisory = self._runner.on_query(query)
            summary: dict[str, np.ndarray] = advisory.get("state_summary", {})
            if not summary:
                return None
            weights = np.zeros(self.n_stacks, dtype=np.float32)
            for vec in summary.values():
                v = np.asarray(vec, dtype=np.float32)
                if v.shape == (self.n_stacks,):
                    weights += v / N_ORTHO
            return weights
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow advisory failed, passthrough: %s", e)
            return None
