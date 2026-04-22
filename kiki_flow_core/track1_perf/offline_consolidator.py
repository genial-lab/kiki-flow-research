"""CLI entrypoint for a single offline T1 consolidation step."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.hooks import AeonAdapter, MoELoraAdapter, RoutingAdapter
from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.modules import AdvectionDiffusion, PhonologicalLoop, ScaffoldingScheduler
from kiki_flow_core.species import MixedCanonicalSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track1_perf.checkpoint import load_checkpoint, save_checkpoint
from kiki_flow_core.track1_perf.eulerian_grid_solver import EulerianGridSolver
from kiki_flow_core.track1_perf.phenomenological_f import T1FreeEnergy

logger = logging.getLogger("kiki_flow.t1")


def _bootstrap_state(
    stack_names: list[str],
    n_grid: int,
    p_theta: np.ndarray,
) -> FlowState:
    species = MixedCanonicalSpecies(stack_names=stack_names)
    names = species.species_names()
    rho_uniform = np.full(n_grid, 1.0 / n_grid)
    return FlowState(
        rho={name: rho_uniform.copy() for name in names},
        P_theta=p_theta,
        mu_curr=rho_uniform.copy(),
        tau=0,
        metadata={"track_id": "T1", "step_id": str(int(time.time()))},
    )


def run_once(
    config: dict[str, Any],
    aeon_fetcher: Callable[[int], list[dict[str, Any]]],
    moe_snapshotter: Callable[[], dict[str, np.ndarray]],
    advisory_publisher: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    stack_names: list[str] = list(config["stack_names"])
    n_grid: int = int(config["n_grid"])
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    aeon = AeonAdapter(fetcher=aeon_fetcher)
    moe = MoELoraAdapter(snapshotter=moe_snapshotter)
    routing = RoutingAdapter(publisher=advisory_publisher)

    latest = ckpt_dir / "latest"
    p_theta = np.concatenate(list(moe.snapshot_stack_states().values()))
    if latest.with_suffix(".safetensors").exists():
        state = load_checkpoint(latest)
    else:
        state = _bootstrap_state(stack_names, n_grid, p_theta)

    species = MixedCanonicalSpecies(stack_names=stack_names)
    x = np.linspace(-2.0, 2.0, n_grid)
    adv_diff = AdvectionDiffusion(species=species, x_grid=x, diffusion=0.005)
    scheduler = ScaffoldingScheduler(h_min=1e-2, h_max=0.1)
    phono = PhonologicalLoop(detector=np.zeros_like, correction_strength=0.05)
    f_t1 = T1FreeEnergy(alpha=1.0, beta=0.1, gamma=0.5, species=species, v_curr=np.zeros(n_grid))
    jko = JKOStep(
        f_functional=f_t1,
        h=0.05,
        support=x.reshape(-1, 1),
        n_inner=10,
        apply_w2_prox=False,
    )

    solver = EulerianGridSolver(species, scheduler, adv_diff, jko, phono)
    new_state = solver.step(state)

    advisory: dict[str, Any] = {
        "tau": new_state.tau,
        "stack_weights": {s: 1.0 / len(stack_names) for s in stack_names},
    }
    routing.publish_advisory(advisory)
    save_checkpoint(new_state, latest)

    _ = aeon.fetch_recent_episodes(window_h=24)

    return {"status": "ok", "tau": new_state.tau, "advisory": advisory}


def main(argv: list[str] | None = None) -> int:
    """CLI: runs one T1 consolidation step with stub hooks (override in prod)."""
    _ = argv  # CLI args not used in v1; stubs passed directly for tests.
    logging.basicConfig(level=logging.INFO)
    stack_names = ["code", "math"]
    cfg: dict[str, Any] = {
        "stack_names": stack_names,
        "n_grid": 256,
        "checkpoint_dir": Path("bench/runs/T1"),
    }

    def _empty_aeon(h: int) -> list[dict[str, Any]]:
        return []

    def _zero_stacks() -> dict[str, np.ndarray]:
        return {name: np.zeros(8) for name in stack_names}

    def _log_advisory(adv: dict[str, Any]) -> None:
        logger.info("advisory: %s", adv)

    manifest = run_once(
        config=cfg,
        aeon_fetcher=_empty_aeon,
        moe_snapshotter=_zero_stacks,
        advisory_publisher=_log_advisory,
    )
    logger.info("T1 consolidation manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
