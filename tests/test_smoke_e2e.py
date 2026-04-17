"""End-to-end smoke test exercising the full kiki-flow core pipeline."""

import io

import numpy as np
import pytest

from kiki_flow_core import (
    AdvectionDiffusion,
    FlowState,
    InvariantViolationError,
    JKOStep,
    Metrics,
    PhonologicalLoop,
    RoutingAdapter,
    ScaffoldingScheduler,
    StructuredLogger,
    ZeroF,
    assert_invariants,
)


def make_initial_state(n: int = 32) -> FlowState:
    rho = np.exp(-0.5 * np.linspace(-2, 2, n) ** 2)
    rho /= rho.sum()
    return FlowState(
        rho={"phono": rho},
        P_theta=np.zeros(8),
        mu_curr=np.full(n, 1.0 / n),
        tau=0,
        metadata={"track_id": "T1", "step_id": "smoke-init"},
    )


def test_e2e_5_steps_smoke() -> None:
    """One full mini-pipeline: 5 consolidation steps, invariants hold, telemetry filled."""
    n = 32
    x = np.linspace(-2, 2, n)
    state = make_initial_state(n)
    assert_invariants(state)

    adv = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.005)
    scheduler = ScaffoldingScheduler(h_min=1e-2, h_max=0.5)
    loop = PhonologicalLoop(detector=lambda out: 0.1 * np.sin(out), correction_strength=0.05)
    jko = JKOStep(f_functional=ZeroF(), h=0.05, support=x.reshape(-1, 1), n_inner=10)

    routing_received: list[dict] = []
    routing = RoutingAdapter(publisher=routing_received.append)
    metrics = Metrics()
    log_buf = io.StringIO()
    logger = StructuredLogger(stream=log_buf)

    for _ in range(5):
        h, _mu_curr = scheduler.next_step(error_profile=np.full(n, 0.3))
        v_field = np.full(n, 0.05)
        source = loop.source_term(rho_phono=state.rho["phono"], output=state.rho["phono"])
        rho_intermediate = adv.step_1d(state.rho["phono"], v_field=v_field, dt=h, source=source)
        state = state.model_copy(update={"rho": {"phono": rho_intermediate}})
        state = jko.step(state)
        assert_invariants(state)
        routing.publish_advisory({"tau": state.tau, "h": h})
        metrics.record(track="T1", metric_name="steps_total", value=1, kind="counter")
        logger.record(track="T1", tau=state.tau, step_phase="full", status="ok", duration_ms=1.0)

    assert state.tau == 5  # noqa: PLR2004
    assert len(routing_received) == 5  # noqa: PLR2004
    assert metrics.snapshot()[("T1", "steps_total")] == 5  # noqa: PLR2004
    assert log_buf.getvalue().count("\n") == 5  # noqa: PLR2004


def test_double_run_deterministic() -> None:
    """Two identical runs with same seed must produce bit-identical state on CPU."""
    np.random.seed(7)
    state1 = make_initial_state(16)
    np.random.seed(7)
    state2 = make_initial_state(16)
    np.testing.assert_array_equal(state1.rho["phono"], state2.rho["phono"])

    x = np.linspace(-1, 1, 16)
    adv = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.001)
    rho_a = state1.rho["phono"].copy()
    rho_b = state2.rho["phono"].copy()
    v = np.full(16, 0.1)
    for _ in range(20):
        rho_a = adv.step_1d(rho_a, v_field=v, dt=0.01)
        rho_b = adv.step_1d(rho_b, v_field=v, dt=0.01)
    np.testing.assert_allclose(rho_a, rho_b, atol=0.0)


def test_invariant_violation_caught() -> None:
    """A bad state raises InvariantViolationError from assert_invariants."""
    bad = FlowState(
        rho={"phono": np.array([0.1, 0.1])},  # mass = 0.2
        P_theta=np.zeros(2),
        mu_curr=np.array([1.0]),
        tau=0,
        metadata={"track_id": "T1"},
    )
    with pytest.raises(InvariantViolationError):
        assert_invariants(bad)
