from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.hooks import RoutingAdapter
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
from kiki_flow_core.track3_deploy.streaming_runner import StreamingRunner


def _write_tiny_weights(
    path: Path, state_dim: int = 16, embed_dim: int = 384, hidden: int = 32
) -> None:
    rng = np.random.default_rng(0)
    tensors = {
        "w1": (rng.standard_normal((state_dim + embed_dim, hidden)) * 0.001).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.standard_normal((hidden, hidden)) * 0.001).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "w3": (rng.standard_normal((hidden, state_dim)) * 0.001).astype(np.float32),
        "b3": np.zeros(state_dim, dtype=np.float32),
    }
    save_file(tensors, str(path))


def make_initial_state(n: int = 4) -> FlowState:
    return FlowState(
        rho={f"{o}:code": np.full(n, 1.0 / n) for o in ["phono", "lex", "syntax", "sem"]},
        P_theta=np.zeros(8),
        mu_curr=np.full(n, 1.0 / n),
        tau=0,
        metadata={"track_id": "T3"},
    )


def test_streaming_runner_on_query_invokes_routing(tmp_path: Path):
    _write_tiny_weights(tmp_path / "w.safetensors", state_dim=16, hidden=32)
    surr = NeuralSurrogate.load(tmp_path / "w.safetensors", state_dim=16, embed_dim=384, hidden=32)
    enc = QueryEncoder(use_stub=True)
    publisher = MagicMock()
    routing = RoutingAdapter(publisher=publisher)
    runner = StreamingRunner(
        surrogate=surr,
        encoder=enc,
        routing_adapter=routing,
        initial_state=make_initial_state(n=4),
    )
    advisory = runner.on_query("hello world")
    assert publisher.called
    assert "tau" in advisory
