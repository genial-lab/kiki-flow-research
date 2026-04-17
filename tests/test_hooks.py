import time

import numpy as np
import pytest

from kiki_flow_core.hooks.aeon_adapter import AeonAdapter, CircuitBreakerOpenError
from kiki_flow_core.hooks.moe_lora_adapter import MoELoraAdapter
from kiki_flow_core.hooks.routing_adapter import RoutingAdapter


def test_aeon_adapter_returns_episodes_when_healthy():
    def fetcher(window_h: int) -> list[dict]:
        return [{"id": 1, "text": "hello"}]

    a = AeonAdapter(fetcher=fetcher)
    episodes = a.fetch_recent_episodes(window_h=24)
    assert episodes == [{"id": 1, "text": "hello"}]


def test_aeon_adapter_circuit_breaker_opens_after_3_failures():
    def failing_fetcher(window_h: int) -> list[dict]:
        raise ConnectionError("offline")

    a = AeonAdapter(fetcher=failing_fetcher, breaker_threshold=3, breaker_cooldown_s=1.0)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            a.fetch_recent_episodes(window_h=24)
    with pytest.raises(CircuitBreakerOpenError):
        a.fetch_recent_episodes(window_h=24)


def test_aeon_adapter_circuit_breaker_resets_after_cooldown():
    state = {"calls": 0}

    def failing_then_ok(window_h: int) -> list[dict]:
        state["calls"] += 1
        if state["calls"] <= 3:  # noqa: PLR2004
            raise ConnectionError("offline")
        return [{"id": state["calls"]}]

    a = AeonAdapter(fetcher=failing_then_ok, breaker_threshold=3, breaker_cooldown_s=0.1)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            a.fetch_recent_episodes(window_h=24)
    time.sleep(0.15)
    out = a.fetch_recent_episodes(window_h=24)
    assert out[0]["id"] == 4  # noqa: PLR2004


def test_moe_lora_adapter_snapshot_returns_dict():
    a = MoELoraAdapter(snapshotter=lambda: {"code": np.zeros(4), "math": np.ones(4)})
    snap = a.snapshot_stack_states()
    assert "code" in snap
    np.testing.assert_array_equal(snap["math"], np.ones(4))


def test_moe_lora_adapter_returns_stack_names():
    a = MoELoraAdapter(snapshotter=lambda: {"code": np.zeros(2), "math": np.zeros(2)})
    assert sorted(a.stack_names()) == ["code", "math"]


def test_routing_adapter_swallows_callback_exceptions():
    """Advisory must never propagate exceptions back to caller."""

    def bad_callback(advisory: dict) -> None:
        raise RuntimeError("router crashed")

    a = RoutingAdapter(publisher=bad_callback)
    a.publish_advisory({"suggested_stack": "code"})  # should not raise


def test_routing_adapter_calls_callback_with_advisory():
    received: list[dict] = []
    a = RoutingAdapter(publisher=received.append)
    advisory = {"suggested_stack": "math", "confidence": 0.8}
    a.publish_advisory(advisory)
    assert received == [advisory]
