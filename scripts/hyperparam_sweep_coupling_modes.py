"""Coupling-mode ablation sweep.

Compares five coupling modes for specialization (mean entropy in bits):

1. ``separable``          -- SeparableEnergy (no J, no Turing)
2. ``symmetric-dell``     -- T2FreeEnergy (Dell YAML) with J_asym zeroed
3. ``dell-full``          -- T2FreeEnergy (Dell YAML), full J, default behaviour
4. ``symmetric-levelt``   -- T2FreeEnergy (Levelt YAML) with J_asym zeroed
5. ``levelt-full``        -- T2FreeEnergy (Levelt YAML), full J

Grid: alpha in {1.0, 5.0, 10.0} x beta in {0.5, 1.0, 2.0} x gamma = 0.0 x 5 modes.
45 (alpha, beta, mode) cells; each averaged over SEEDS = [0, 1, 2, 3, 4].

Modelled on scripts/hyperparam_sweep_dense.py (same grid=16, n_fast=200,
n_slow=30, 500 particles, MLXParticleSimulator + MultiscaleLoop + JKOStep).
Writes results incrementally to paper/hyperparam_sweep_coupling_modes.json so
partial progress is recoverable. Produces 5-panel heatmap and grouped bar
plot + short Markdown summary in paper/figures/ and paper/.

Invocation:
    uv run python scripts/hyperparam_sweep_coupling_modes.py \\
        --output paper/hyperparam_sweep_coupling_modes.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kiki_flow_core.master_equation import JKOStep  # noqa: E402
from kiki_flow_core.species import CanonicalSpecies  # noqa: E402
from kiki_flow_core.track2_paper.mlx_particle_simulator import MLXParticleSimulator  # noqa: E402
from kiki_flow_core.track2_paper.multiscale_loop import MultiscaleLoop  # noqa: E402
from kiki_flow_core.track2_paper.paper_f import T2FreeEnergy  # noqa: E402
from kiki_flow_core.track2_paper.paper_f_separable import SeparableEnergy  # noqa: E402
from kiki_flow_core.wasserstein_ops import w2_distance  # noqa: E402

# ---- Sweep configuration --------------------------------------------------

GRID = 16
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [1.0, 5.0, 10.0]
BETAS = [0.5, 1.0, 2.0]
GAMMA = 0.0  # Turing off for this ablation.

MODES = [
    "separable",
    "symmetric-dell",
    "dell-full",
    "symmetric-levelt",
    "levelt-full",
]

# max_entropy := log2(n_species * grid) per the task spec.
# With 4 species x 16 bins, the upper bound is log2(64) = 6.0 bits when the
# aggregated (rho stacked) distribution is uniform. Each individual rho_i is
# capped at log2(grid) = 4 bits.
N_SPECIES = 4
MAX_ENTROPY_BITS = float(np.log2(N_SPECIES * GRID))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---- Energy construction -------------------------------------------------


def _make_potentials(alpha: float, x: np.ndarray) -> dict[str, np.ndarray]:
    """Match scripts/hyperparam_sweep_dense.py potentials."""
    return {
        "phono": alpha * (x + 1.5),
        "lex": alpha * (x - 1.5),
        "syntax": alpha * np.abs(x),
        "sem": -alpha * np.abs(x),
    }


def _build_energy(
    mode: str,
    species: CanonicalSpecies,
    potentials: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    gamma: float,
) -> SeparableEnergy | T2FreeEnergy:
    if mode == "separable":
        return SeparableEnergy(species=species, potentials=potentials, prior=prior)
    energy = T2FreeEnergy(
        species=species,
        potentials=potentials,
        prior=prior,
        turing_strength=gamma * 0.1,
    )
    if mode.startswith("symmetric-"):
        # Kill the non-conservative drift (drift only shows up in
        # coupling_drive, not in value(), but we zero it anyway so that
        # any downstream helper that pulls J_asym sees a zero matrix).
        energy._J_asym = np.zeros_like(energy._J_asym)
        # Legacy helpers that read _coupling should only see the symmetric
        # part, per the task spec.
        energy._coupling = energy._J_sym.copy()
    return energy


def _species_for_mode(mode: str) -> CanonicalSpecies:
    if "levelt" in mode:
        return CanonicalSpecies(coupling_variant="levelt")
    # separable and dell-* use the dell YAML (J gets zeroed or ignored for
    # separable; doesn't matter which YAML per task spec).
    return CanonicalSpecies(coupling_variant="dell")


# ---- One-shot solve -------------------------------------------------------


def run_one(
    alpha: float,
    beta: float,
    gamma: float,
    mode: str,
    seed: int,
    apply_asym_drift: bool = False,
) -> dict[str, Any]:
    species = _species_for_mode(mode)
    names = species.species_names()
    support = np.linspace(-2, 2, GRID).reshape(-1, 1)
    x = support[:, 0]
    potentials = _make_potentials(alpha, x)
    prior = {n: np.full(GRID, 1.0 / GRID) for n in names}
    f_func = _build_energy(mode, species, potentials, prior, gamma)
    n_inner = max(1, int(10 * beta)) if beta > 0 else 1
    jko = JKOStep(
        f_functional=f_func,
        h=0.05,
        support=support,
        n_inner=n_inner,
        apply_w2_prox=False,
    )
    sim = MLXParticleSimulator(species=species, n_particles=500, latent_dim=2, seed=seed)
    # For the "*-full" modes, apply an operator-splitting step of the
    # non-conservative antisymmetric drift b_asym = J_asym @ rho after each JKO
    # step. "symmetric-*" and "separable" have J_asym = 0 (or no J at all), so
    # the hook would be the identity map and is skipped for clarity.
    post_jko_hook = None
    if (
        apply_asym_drift
        and mode in ("dell-full", "levelt-full")
        and isinstance(f_func, T2FreeEnergy)
    ):
        energy_for_hook = f_func

        def post_jko_hook(state, _e=energy_for_hook):  # noqa: E501
            return _e.apply_drift_splitting(state, h_drift=0.01)

    loop = MultiscaleLoop(
        sim=sim,
        jko=jko,
        n_fast=200,
        n_slow=30,
        support=support,
        post_jko_hook=post_jko_hook,
    )
    manifest = loop.run(seed=seed)
    final = manifest["trajectory"][-1]

    # Per-species entropies (bits).
    per_species_entropy = {}
    rhos = []
    for n in names:
        rho = np.clip(np.asarray(final.rho[n]), 1e-12, None)
        per_species_entropy[n] = float(-(rho * np.log2(rho)).sum())
        rhos.append(rho)

    # Aggregated entropy over the stacked (rho_1, ..., rho_K) / K distribution:
    # this gives a joint-specialization entropy that can reach
    # log2(N_SPECIES * GRID).
    stacked = np.concatenate([r for r in rhos]) / float(N_SPECIES)
    stacked = np.clip(stacked, 1e-12, None)
    joint_entropy = float(-(stacked * np.log2(stacked)).sum())

    # "mean_entropy" == mean of per-species entropies (interpretation matching
    # the task deliverable "color = mean_entropy_bits"). Max per-species H
    # is log2(GRID) = 4 bits, so mean_entropy is bounded by 4 bits.
    mean_entropy = float(np.mean(list(per_species_entropy.values())))
    # specialization_gap_bits using the task-spec max_entropy of
    # log2(n_species * GRID). This gives a larger headroom; keeping the
    # joint_entropy reported too for cross-check.
    gap = MAX_ENTROPY_BITS - mean_entropy

    # W2 distances: to uniform prior and pairwise between species (proxy for
    # "final W2 distances to fixed points" -- the natural fixed points for
    # this setup are the per-species potential minima, so we also add W2
    # from each rho to a delta at argmin of that species' potential).
    w2_to_uniform = {}
    w2_to_fixed = {}
    # Use a slightly larger regularization (0.02) with 5000 iterations for
    # the fixed-point-to-rho distances. The delta-like fixed points produce
    # ill-conditioned marginals at epsilon=0.005, and the CLAUDE.md policy
    # is to legitimately widen epsilon/n_iter rather than silence warnings.
    w2_eps = 0.05
    w2_iter = 10000
    for n in names:
        rho = np.asarray(final.rho[n])
        rho = rho / rho.sum()
        uniform = np.full(GRID, 1.0 / GRID)
        w2_to_uniform[n] = float(w2_distance(rho, uniform, support, epsilon=w2_eps, n_iter=w2_iter))
        # Fixed point: delta at argmin of V_i on the grid (the location the
        # potential drive pulls the particles to at beta->infty). Smoothed
        # by a tiny mass floor to keep Sinkhorn well-conditioned.
        v_i = potentials[n]
        idx = int(np.argmin(v_i))
        delta = np.full(GRID, 1e-6)
        delta[idx] = 1.0 - (GRID - 1) * 1e-6
        delta = delta / delta.sum()
        w2_to_fixed[n] = float(w2_distance(rho, delta, support, epsilon=w2_eps, n_iter=w2_iter))

    return {
        "seed": seed,
        "per_species_entropy_bits": per_species_entropy,
        "mean_entropy_bits": mean_entropy,
        "joint_entropy_bits": joint_entropy,
        "specialization_gap_bits": gap,
        "w2_to_uniform": w2_to_uniform,
        "w2_to_fixed_point": w2_to_fixed,
    }


# ---- Incremental writer --------------------------------------------------


class IncrementalWriter:
    """Write the JSON output after every completed row so crashes are safe."""

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        self.path = path
        self.meta = meta
        self.rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        payload = {
            **self.meta,
            "n_rows_written": len(self.rows),
            "results": self.rows,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)


# ---- Plotting ------------------------------------------------------------


def plot_heatmaps(rows: list[dict[str, Any]], out_path: Path) -> None:
    modes = MODES
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 3.6), sharey=True)
    vmin = min(r["mean_entropy_bits"] for r in rows)
    vmax = max(r["mean_entropy_bits"] for r in rows)
    for ax, mode in zip(axes, modes, strict=True):
        grid = np.full((len(BETAS), len(ALPHAS)), np.nan)
        for i, b in enumerate(BETAS):
            for j, a in enumerate(ALPHAS):
                match = [
                    r
                    for r in rows
                    if r["alpha"] == a and r["beta"] == b and r["coupling_mode"] == mode
                ]
                if match:
                    grid[i, j] = match[0]["mean_entropy_bits"]
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f"{a:.0f}" for a in ALPHAS])
        ax.set_yticks(range(len(BETAS)))
        ax.set_yticklabels([f"{b:.1f}" for b in BETAS])
        ax.set_xlabel(r"$\alpha$ (potential)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\beta$ (JKO inner steps)")
        ax.set_title(mode, fontsize=10)
        # Annotate cells.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if val < 0.5 * (vmin + vmax) else "black",
                        fontsize=8,
                    )
    fig.suptitle(
        rf"Mean per-species entropy (bits) by coupling mode, $\gamma=0$ "
        rf"(max=log$_2$({GRID})=4.00)",
        fontsize=11,
    )
    fig.colorbar(im, ax=axes, label="bits", shrink=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_barplot(rows: list[dict[str, Any]], out_path: Path) -> tuple[float, float]:
    """Grouped bar plot at the (alpha, beta) that minimises mean_entropy in dell-full."""
    dell_full = [r for r in rows if r["coupling_mode"] == "dell-full"]
    best = min(dell_full, key=lambda r: r["mean_entropy_bits"])
    alpha_star, beta_star = best["alpha"], best["beta"]
    at_best = {
        r["coupling_mode"]: r for r in rows if r["alpha"] == alpha_star and r["beta"] == beta_star
    }
    modes = MODES
    means = [at_best[m]["mean_entropy_bits"] for m in modes]
    stds = [
        float(np.std([sr["mean_entropy_bits"] for sr in at_best[m]["seed_results"]])) for m in modes
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xpos = np.arange(len(modes))
    bars = ax.bar(
        xpos,
        means,
        yerr=stds,
        capsize=4,
        color=["#888", "#4a7", "#174", "#4ad", "#14a"],
        edgecolor="black",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel("Mean entropy (bits)")
    ax.set_title(
        rf"Coupling-mode comparison at $\alpha={alpha_star}$, $\beta={beta_star}$, "
        rf"$\gamma=0$ (lower = more specialized)"
    )
    ax.axhline(float(np.log2(GRID)), color="red", linestyle="--", alpha=0.6, label="max=4.0 bits")
    ax.legend()
    for bar, m in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return alpha_star, beta_star


def write_summary(
    rows: list[dict[str, Any]],
    alpha_star: float,
    beta_star: float,
    out_path: Path,
    total_runtime: float,
) -> None:
    dell_full = next(
        r
        for r in rows
        if r["coupling_mode"] == "dell-full" and r["alpha"] == alpha_star and r["beta"] == beta_star
    )
    levelt_full = next(
        r
        for r in rows
        if r["coupling_mode"] == "levelt-full"
        and r["alpha"] == alpha_star
        and r["beta"] == beta_star
    )
    separable = next(
        r
        for r in rows
        if r["coupling_mode"] == "separable" and r["alpha"] == alpha_star and r["beta"] == beta_star
    )
    sym_dell = next(
        r
        for r in rows
        if r["coupling_mode"] == "symmetric-dell"
        and r["alpha"] == alpha_star
        and r["beta"] == beta_star
    )
    sym_lev = next(
        r
        for r in rows
        if r["coupling_mode"] == "symmetric-levelt"
        and r["alpha"] == alpha_star
        and r["beta"] == beta_star
    )
    md = [
        "# Coupling-mode ablation summary",
        "",
        f"Sweep: {len(ALPHAS) * len(BETAS) * len(MODES)} cells "
        f"({len(ALPHAS)} alpha x {len(BETAS)} beta x {len(MODES)} modes), "
        f"{len(SEEDS)} seeds each.",
        f"Runtime: {total_runtime:.1f} s.  "
        f"Best (lowest H) dell-full cell: alpha={alpha_star}, beta={beta_star}.",
        "",
        "## Mean per-species entropy (bits) at best dell-full cell",
        "",
        "| mode | mean H | std |",
        "| --- | --- | --- |",
    ]
    for m, r in [
        ("separable", separable),
        ("symmetric-dell", sym_dell),
        ("dell-full", dell_full),
        ("symmetric-levelt", sym_lev),
        ("levelt-full", levelt_full),
    ]:
        std = float(np.std([sr["mean_entropy_bits"] for sr in r["seed_results"]]))
        md.append(f"| {m} | {r['mean_entropy_bits']:.4f} | {std:.4f} |")
    md.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- separable vs dell-full: Delta H = "
            f"{separable['mean_entropy_bits'] - dell_full['mean_entropy_bits']:+.4f} bits "
            f"({100 * (separable['mean_entropy_bits'] - dell_full['mean_entropy_bits']) / max(dell_full['mean_entropy_bits'], 1e-9):+.2f}%).",  # noqa: E501
            f"- separable vs levelt-full: Delta H = "
            f"{separable['mean_entropy_bits'] - levelt_full['mean_entropy_bits']:+.4f} bits.",
            f"- symmetric-dell vs dell-full: Delta H = "
            f"{sym_dell['mean_entropy_bits'] - dell_full['mean_entropy_bits']:+.4f} bits "
            "(tests whether J_asym drift matters in the solver path).",
            f"- symmetric-levelt vs levelt-full: Delta H = "
            f"{sym_lev['mean_entropy_bits'] - levelt_full['mean_entropy_bits']:+.4f} bits.",
            f"- dell-full vs levelt-full: Delta H = "
            f"{dell_full['mean_entropy_bits'] - levelt_full['mean_entropy_bits']:+.4f} bits "
            "(tests whether coupling topology matters).",
            "",
            "## Notes",
            "",
            "- `T2FreeEnergy.value()` in `paper_f.py` routes the scalar coupling energy "
            "through `J_sym` only (the antisymmetric contraction vanishes identically), so "
            "the JKO gradient path sees `J_sym` regardless of whether `J_asym` is zeroed. "
            "`symmetric-*` and `*-full` modes are therefore expected to coincide to within "
            "numerical noise on this solver -- the ablation primarily isolates (separable "
            "vs coupled) and (Dell vs Levelt topology).",
            "- `max_entropy = log2(n_species * grid) = "
            f"{MAX_ENTROPY_BITS:.4f}` bits per task spec; per-species entropy is bounded by "
            f"log2(grid) = {float(np.log2(GRID)):.4f} bits.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n")


# ---- Driver --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/hyperparam_sweep_coupling_modes.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/coupling_modes_summary.md"),
    )
    parser.add_argument(
        "--apply-asym-drift",
        action="store_true",
        help=(
            "If set, the dell-full and levelt-full modes activate the "
            "matrix-exponential splitting step of the antisymmetric drift "
            "b_asym = J_asym @ rho (via T2FreeEnergy.apply_drift_splitting). "
            "Default off preserves numerical reproducibility of the "
            "committed paper/hyperparam_sweep_coupling_modes.json; set to "
            "true to exercise the feature experimentally."
        ),
    )
    args = parser.parse_args()

    sha = git_sha()
    t_global = time.time()
    meta = {
        "grid": GRID,
        "seeds": SEEDS,
        "alphas": ALPHAS,
        "betas": BETAS,
        "gamma": GAMMA,
        "modes": MODES,
        "max_entropy_bits": MAX_ENTROPY_BITS,
        "git_sha": sha,
    }
    writer = IncrementalWriter(args.output, meta)

    combos = list(itertools.product(ALPHAS, BETAS, MODES))
    n_cells = len(combos)
    for idx, (alpha, beta, mode) in enumerate(combos, start=1):
        t0 = time.time()
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            res = run_one(alpha, beta, GAMMA, mode, seed, apply_asym_drift=args.apply_asym_drift)
            per_seed.append(res)
        mean_entropy = float(np.mean([r["mean_entropy_bits"] for r in per_seed]))
        gap = float(np.mean([r["specialization_gap_bits"] for r in per_seed]))
        row = {
            "alpha": alpha,
            "beta": beta,
            "gamma": GAMMA,
            "coupling_mode": mode,
            "mean_entropy_bits": mean_entropy,
            "specialization_gap_bits": gap,
            "seed_results": per_seed,
            "runtime_sec": time.time() - t0,
            "git_sha": sha,
        }
        writer.append(row)
        elapsed = time.time() - t_global
        print(
            f"[{idx}/{n_cells}] alpha={alpha} beta={beta} mode={mode:<18}"
            f" -> H_bits={mean_entropy:.4f}  (elapsed {elapsed:.1f}s)"
        )

    # Post-processing.
    rows = writer.rows
    heatmap_path = args.figures_dir / "coupling_modes_heatmap.png"
    plot_heatmaps(rows, heatmap_path)
    barplot_path = args.figures_dir / "coupling_modes_barplot.png"
    alpha_star, beta_star = plot_barplot(rows, barplot_path)
    total_runtime = time.time() - t_global
    write_summary(rows, alpha_star, beta_star, args.summary, total_runtime)

    print(f"\nWrote heatmap:  {heatmap_path}")
    print(f"Wrote barplot:  {barplot_path}")
    print(f"Wrote summary:  {args.summary}")
    print(f"Wrote results:  {args.output}")
    print(f"Total runtime:  {total_runtime:.1f}s")
    print(f"git_sha:        {sha}")


if __name__ == "__main__":
    main()
