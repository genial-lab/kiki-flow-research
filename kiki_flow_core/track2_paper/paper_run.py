"""Top-level driver: run 5 seeds, produce figures, aggregate stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from safetensors.numpy import save_file

from kiki_flow_core.master_equation import JKOStep
from kiki_flow_core.species import OrthoSpecies
from kiki_flow_core.state import FlowState
from kiki_flow_core.track2_paper.figures.continual_learning_gap import (
    make_continual_learning_gap,
)
from kiki_flow_core.track2_paper.figures.f_decay_curves import make_f_decay_curves
from kiki_flow_core.track2_paper.figures.kl_vs_epsilon import make_kl_vs_epsilon
from kiki_flow_core.track2_paper.figures.phase_portrait import make_phase_portrait
from kiki_flow_core.track2_paper.figures.turing_patterns import make_turing_patterns
from kiki_flow_core.track2_paper.full_jko_solver import FullJKOSolver, MLXFullJKOSolver
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy
from kiki_flow_core.track2_paper.particle_simulator import ParticleSimulator

SinkhornBackend = Literal["pot", "mlx"]


def _persist_trajectory(trajectory: list[FlowState], out_path: Path) -> None:
    """Save a trajectory list of FlowStates as safetensors (rho arrays only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for tau_idx, state in enumerate(trajectory):
        for name, rho in state.rho.items():
            tensors[f"tau{tau_idx:03d}::{name}"] = np.asarray(rho, dtype=np.float32)
    save_file(tensors, str(out_path))


def _aggregate_rho_stats(trajectories: list[list[FlowState]]) -> dict[str, Any]:
    """Aggregate informative stats per species at the final tau: peak amplitude
    (max of rho), Shannon entropy (bits, lower = more concentrated), and the
    location of the peak on the grid index."""
    names = list(trajectories[0][-1].rho.keys())
    peak: dict[str, list[float]] = {n: [] for n in names}
    entropy_bits: dict[str, list[float]] = {n: [] for n in names}
    peak_idx: dict[str, list[int]] = {n: [] for n in names}
    for traj in trajectories:
        final = traj[-1]
        for n in names:
            rho = np.asarray(final.rho[n])
            rho_safe = np.clip(rho, 1e-12, None)
            peak[n].append(float(rho.max()))
            entropy_bits[n].append(float(-(rho_safe * np.log2(rho_safe)).sum()))
            peak_idx[n].append(int(np.argmax(rho)))
    return {
        n: {
            "peak_mean": float(np.mean(peak[n])),
            "peak_std": float(np.std(peak[n])),
            "entropy_bits_mean": float(np.mean(entropy_bits[n])),
            "entropy_bits_std": float(np.std(entropy_bits[n])),
            "peak_idx_mean": float(np.mean(peak_idx[n])),
            "peak_idx_std": float(np.std(peak_idx[n])),
        }
        for n in names
    }


def run_paper(
    seeds: list[int],
    n_particles: int = 10_000,
    n_fast: int = 1000,
    n_slow: int = 100,
    grid_size: int = 64,
    out_dir: Path = Path("paper"),
    use_mlx: bool = True,
    use_w2_prox: bool = False,
    sinkhorn_backend: SinkhornBackend = "pot",
    save_trajectories: bool = False,
    make_all_figures: bool = False,
) -> dict[str, Any]:
    """Run N seeds of the T2 multiscale loop, producing figures and stats.

    Parameters
    ----------
    use_mlx : bool (default True)
        Use MLXParticleSimulator (Apple Silicon Metal) instead of numpy.
    use_w2_prox : bool (default False)
        Use FullJKOSolver with Wasserstein proximal step (rigorous but ~55x slower).
        Default False uses plain JKOStep (projected gradient descent only).
    save_trajectories : bool (default False)
        Persist each seed's trajectory as safetensors for post-hoc figure
        generation or ablations.
    make_all_figures : bool (default False)
        Produce fig1-5 using the first seed's trajectory (requires
        save_trajectories implicitly). fig4/fig5 use synthetic input
        unless a real KL-sweep or CL benchmark is plumbed in.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    species = OrthoSpecies()
    names = species.species_names()
    support = np.linspace(-2, 2, grid_size).reshape(-1, 1)
    potentials = {n: np.zeros(grid_size) for n in names}
    prior = {n: np.full(grid_size, 1.0 / grid_size) for n in names}
    f_func = T2FreeEnergy(species=species, potentials=potentials, prior=prior, turing_strength=0.1)

    if use_w2_prox:
        if sinkhorn_backend == "mlx":
            jko: JKOStep = MLXFullJKOSolver(
                f_functional=f_func, h=0.05, support=support, epsilon=0.01
            )
        else:
            jko = FullJKOSolver(f_functional=f_func, h=0.05, support=support, epsilon=0.01)
    else:
        jko = JKOStep(f_functional=f_func, h=0.05, support=support, n_inner=10, apply_w2_prox=False)

    per_seed: list[dict[str, Any]] = []
    trajectories: list[list[FlowState]] = []
    for seed in seeds:
        if use_mlx:
            sim: Any = MLXParticleSimulator(
                species=species, n_particles=n_particles, latent_dim=2, seed=seed
            )
        else:
            sim = ParticleSimulator(
                species=species, n_particles=n_particles, latent_dim=2, seed=seed
            )
        loop = MultiscaleLoop(sim=sim, jko=jko, n_fast=n_fast, n_slow=n_slow, support=support)
        manifest = loop.run(seed=seed)
        make_phase_portrait(
            manifest["trajectory"],
            out_dir / "figures",
            filename=f"fig1_phase_seed{seed}",
        )
        if save_trajectories or make_all_figures:
            _persist_trajectory(
                manifest["trajectory"], out_dir / "trajectories" / f"seed{seed}.safetensors"
            )
            trajectories.append(manifest["trajectory"])
        per_seed.append({"seed": seed, "n_slow_completed": manifest["n_slow_completed"]})

    stats: dict[str, Any] = {
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "backend": {
            "simulator": "mlx" if use_mlx else "numpy",
            "w2_prox": use_w2_prox,
            "sinkhorn_backend": sinkhorn_backend if use_w2_prox else None,
        },
    }

    if make_all_figures and trajectories:
        traj0 = trajectories[0]
        make_f_decay_curves(traj0, f_functional=f_func, out_dir=out_dir / "figures")
        make_turing_patterns(traj0, out_dir=out_dir / "figures")
        # KL-vs-eps: synthetic placeholder (real sweep elsewhere)
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
        kl_values = [1.2, 0.45, 0.12, 0.018, 0.003]
        make_kl_vs_epsilon(epsilons, kl_values, out_dir=out_dir / "figures")
        # CL gap: synthetic placeholder
        make_continual_learning_gap(
            tasks=["phonology", "lexicon", "syntax"],
            with_consolidation=[0.87, 0.81, 0.78],
            without_consolidation=[0.72, 0.64, 0.61],
            out_dir=out_dir / "figures",
        )
        stats["aggregate"] = _aggregate_rho_stats(trajectories)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
