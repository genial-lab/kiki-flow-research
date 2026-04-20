"""Streaming inference: on_query -> encode -> surrogate forward -> advisory."""

from __future__ import annotations

from typing import Any

import numpy as np

from kiki_flow_core.hooks import RoutingAdapter
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
from kiki_flow_core.track3_deploy.state_projection import flatten, unflatten


class StreamingRunner:
    """Run a single surrogate inference per user query and publish an advisory."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        encoder: QueryEncoder,
        routing_adapter: RoutingAdapter,
        initial_state: FlowState,
    ) -> None:
        self.surrogate = surrogate
        self.encoder = encoder
        self.routing = routing_adapter
        self.state = initial_state

    def on_query(self, query: str) -> dict[str, Any]:
        query_embed = self.encoder.encode(query)
        state_flat = flatten(self.state).astype(np.float32)
        if state_flat.size != self.surrogate.state_dim:
            return {"tau": self.state.tau, "note": "state_dim mismatch; advisory passthrough"}
        delta = self.surrogate.forward(state_flat, query_embed)
        if not np.isfinite(delta).all():
            return {"tau": self.state.tau, "note": "surrogate produced NaN; advisory suppressed"}
        new_flat = np.clip(state_flat + delta, 1e-6, None)
        new_state = unflatten(new_flat, reference=self.state)
        for k in new_state.rho:
            s = new_state.rho[k].sum()
            if s > 0:
                new_state.rho[k] = new_state.rho[k] / s
        self.state = new_state.model_copy(update={"tau": self.state.tau + 1})
        advisory: dict[str, Any] = {
            "tau": self.state.tau,
            "state_summary": {k: v.astype(np.float32).copy() for k, v in self.state.rho.items()},
        }
        self.routing.publish_advisory(advisory)
        return advisory
