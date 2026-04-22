# Changelog

All notable releases of `kiki-flow-research`. Dates in YYYY-MM-DD.

## [Unreleased] — 2026-04-22 (post v0.7, refactor wave)

### Added
- `T2FreeEnergy` now splits its coupling matrix into symmetric and
  antisymmetric parts at init time (`_J_sym`, `_J_asym`). New helpers
  `_grad_conservative(rhos)`, `_drift_nonconservative(rhos)`, and
  public `coupling_drive(rhos)` isolate the conservative gradient
  piece (from `J_sym`) from the non-conservative drift (from
  `J_asym`). Rationale: the Levelt-Baddeley coupling is 31 %
  asymmetric in Frobenius (`||J_asym||_F / ||J||_F = 0.314`), so the
  historical `Σ J[i,k] ⟨ρ_i, ρ_k⟩` term is the gradient of a scalar
  only through its symmetric part; the antisymmetric part is a pure
  non-conservative drift that cannot arise from any scalar
  functional.
- `SeparableEnergy` ablation baseline in
  `kiki_flow_core/track2_paper/paper_f_separable.py`. Implements
  `F_sep(ρ) = Σ_i ⟨ρ_i, V_i⟩ + Σ_i KL(ρ_i ‖ π_i)` with zero
  inter-species coupling and zero Turing cross-diffusion. Used to
  quantify the fraction of specialization attributable to the J
  coupling versus the per-species potential/entropy machinery alone.
- `CanonicalSpecies.coupling_variant` constructor argument
  (`"dell"` default, `"levelt"` alternative). The historical YAML
  carried a Dell 1986-style matrix (direct `sem → lex = 0.60`
  bypass), not Levelt 1999; the new `levelt_baddeley_coupling.yaml`
  reinstates the syntactic detour (`sem → syn = 0.45`,
  `syn → lex = 0.40`, `sem → lex` direct reduced to `0.10`). Both
  variants preserve the Baddeley phonological backedge
  (`phono → lex = 0.30`). Default `"dell"` keeps existing sweeps
  numerically unchanged.
- `experiments/check_lambda_monotonicity.py` — numerical check of
  the Conger-Hoffmann-Mazumdar-Ratliff 2025 λ-monotonicity condition
  (arXiv 2506.22947). Fits `log d(t) ~ log d(0) - 2 λ̂ · t` on the
  Sinkhorn-divergence trajectories of paired random initial
  conditions. Empirical: `λ̂ = 67.79 ± 3.78` (std/mean 5.6 %) across
  3 seed pairs at `h = 0.002`, `n_inner = 1`; classification
  `λ-monotone`. Supports the paper's exponential-convergence claim
  via Conger et al. Theorem 3.4.
- New tests: `tests/test_j_split_identity.py` (4),
  `tests/track2_paper/test_separable_energy.py` (3),
  `tests/test_coupling_variants.py` (4 parametrized),
  `tests/test_lambda_monotonicity_shape.py` (1 smoke).
- `PERFORMANCE.md` — single-page consolidation of every measured
  performance and quality number reported in the paper, with source
  files and reproduction commands for each entry.
- `.github/workflows/reproducibility.yml` — weekly schedule
  (Monday 07:00 UTC) + `workflow_dispatch` re-runs the deterministic
  scripts and verifies structural keys on the output JSONs.
- Tests: `tests/track2_paper/test_mlx_full_jko_solver.py` (3),
  `tests/test_species_hybrid_edges.py` (4),
  `tests/test_module_error_paths.py` (5). Exercises MLXFullJKOSolver,
  MixedCanonicalSpecies init branches, and error paths in scheduler /
  phonological loop / advection-diffusion.

### Renamed
- `OrthoSpecies` → `CanonicalSpecies`. The "Ortho" prefix suggested
  an L² orthogonality that was never imposed; the class is simply a
  canonical four-species decomposition (phono, lex, syntax, sem) on
  a shared spatial grid. The rename is a clean cut, no
  backward-compatibility alias.
- `HybridSpecies` → `MixedCanonicalSpecies`. The `P[canonical, stack]`
  projection is row-stochastic (each canonical species is a convex
  combination of stacks), not an orthogonal decomposition.
- File `kiki_flow_core/species/ortho_baddeley_levelt.py` →
  `kiki_flow_core/species/canonical_species.py`.
- File `kiki_flow_core/species/hybrid_ortho_stacks.py` →
  `kiki_flow_core/species/mixed_canonical_stacks.py`.
- Historical YAML renamed
  `levelt_baddeley_coupling.yaml` → `dell_baddeley_coupling.yaml`
  (its matrix is phenomenologically Dell 1986); the new
  `levelt_baddeley_coupling.yaml` carries the canonical Levelt matrix.
- Test file `tests/test_species_ortho.py` → `tests/test_species_canonical.py`.

### Changed
- `T2FreeEnergy.value()` routes the bilinear coupling through
  `J_sym`; scalar output is bit-for-bit unchanged because
  `Σ_{i,k} J_asym[i,k] ⟨ρ_i, ρ_k⟩ = 0` identically.
- Test suite: 103 → 256 (+12 from this refactor wave, the remainder
  from prior [Unreleased] and v0.7 additions). Coverage: 93 → 94 %.
- Removed `README.md.backup` (accidentally committed temp file).

## [paper-v0.7-draft] — 2026-04-18

### Added
- `scripts/fast_vs_rigorous_kl.py` — measures KL divergence between
  fast-path and rigorous-path final distributions on matched seeds.
- `paper/fast_vs_rigorous_kl.json` — records KL per seed + aggregate.

### Changed
- Paper §5.2 replaces the qualitative "trajectories are qualitatively
  similar" with a measured mean KL of $6.4 \times 10^{-4}$ bits (5
  seeds, max $9.4 \times 10^{-4}$), confirming fast path is within
  $\sim 0.06\%$ of rigorous path.

## [paper-v0.6-draft] — 2026-04-18

### Added
- `scripts/cl_benchmark_ewc.py` with `EWCTaskF` class.
- Analytical Fisher diagonal $F_i = 2 + \lambda_{\text{KL}} / \rho_i$.
- `paper/cl_benchmark_ewc.json`; `paper/figures/fig5_cl_gap_ewc.{png,pdf}`.

### Changed
- Paper §5 CL section rewritten: 3-arm comparison (without / naive /
  EWC). EWC restores task2 $60\times$ vs naive.
- Abstract adds a fourth claim (iv) on the stability/plasticity
  trade-off and Fisher resolution.

## [paper-v0.5-draft] — 2026-04-18

### Added
- `scripts/hyperparam_sweep_dense.py` — 72-config sweep.
- `kiki_flow_core/track2_paper/mlx_wasserstein.py` `mlx_prox_w2`.
- `kiki_flow_core/track2_paper/full_jko_solver.py` `MLXFullJKOSolver`.
- `JKOStep` strategy-pattern `prox_fn` parameter for MLX/POT switching.
- `sinkhorn_backend = Literal['pot', 'mlx']` flag on `run_paper`.
- `tests/track2_paper/test_mlx_prox_w2.py` (5 tests incl. POT parity).
- `paper/figures/fig6_heatmap.{png,pdf}` entropy landscape.
- `bench/sinkhorn_backend_bench.jsonl` with measured $5.13\times$
  MLX-over-POT speedup on GrosMac M5.

### Changed
- Best specialization point moves to $(\alpha, \beta, \gamma) = (10, 1, 0)$
  with gap $2.04$ bits (51 % below max) vs $1.40$ bits in the coarse
  sweep.
- Abstract and experiments updated; issues #3 closed, #2 fully closed.

## [paper-v0.4-draft] — 2026-04-18

First releasable paper with measured data on every figure.

### Added
- `scripts/cl_benchmark.py` — 3-task distributional continual-learning
  proxy, 5 seeds, produces the real fig5.
- `scripts/epsilon_sweep.py` — measured Sinkhorn entropic bias across
  5 epsilon values on a smooth peaked prox problem.
- `scripts/hyperparam_sweep.py` — 27-configuration sweep over
  `(alpha, beta, gamma)` surfacing the best specialization point.
- `scripts/dump_hybrid_pairs.py` — Mode-A trajectory generator at
  `state_dim=128` for the T3 surrogate retrain.
- `kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors` —
  surrogate matched to a 32-stack × 4-ortho micro-kiki state.
- `kiki_flow_core/track2_paper/mlx_particle_simulator.py` and
  `mlx_wasserstein.py` — Metal-backed alternatives to the numpy/POT
  defaults (simulator gives 2.6× speedup; Sinkhorn is scaffolded).
- `save_trajectories` and `make_all_figures` flags on `run_paper`.
- `_aggregate_rho_stats` reporting peak amplitude, Shannon entropy,
  and peak-index statistics per species across seeds.
- Five figure generators invoked end-to-end (fig1 phase portraits,
  fig2 F-decay, fig3 Turing profiles, fig4 KL vs epsilon, fig5 CL gap).
- Shared matplotlib `rcParams` for consistent serif 11 pt paper figures.
- `docs/superpowers/patches/` — two unified-diff patches and a
  reference implementation for micro-kiki integration.
- `paper/main.tex` + compiled `main.pdf` (~480 KB) with four related-
  work paragraphs, honest mixed-sign findings, reproducibility
  statement, acknowledgments.
- `bench/T3_latency.jsonl` — SLO latency records for both v0.1 and
  v0.2-d128 surrogate variants.
- Public `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`.

### Changed
- Abstract rewritten twice for honesty: first to remove unverified
  overclaims, then again to report the measured specialization gap.
- Related-work section expanded from four 1-sentence citations to
  four paragraphs covering psycholinguistics, Wasserstein flows,
  continual learning, and neuromorphic mean-field inference.
- Visibility flipped to public on 2026-04-18.

### Known limitations
- fig5 is a distributional proxy, not a full LLM continual-learning
  benchmark (issue #1).
- Hyperparameter sweep is coarse; EWC-style refinement pending
  (issue #2).
- Rigorous JKO path is CPU-bound via POT (issue #3).
- End-to-end micro-kiki integration still requires a small
  upstream refactor (issue #4).

## [paper-v0.3-draft] — 2026-04-17

- Measured Sinkhorn epsilon sweep (replaces synthetic fig4).
- 27-config hyperparameter sweep surfacing the specialization regime.
- Abstract updated with quantitative specialization finding.

## [paper-v0.2-draft] — 2026-04-17

- Five figures (two real, two synthetic placeholders, one panel) +
  aggregate rho statistics (entropy, peak, peak-index).
- Related-work section expanded.
- Trajectory persistence in `paper_run.py`.
- Honest abstract restatement + reproducibility statement +
  acknowledgments paragraph.

## [paper-v0.1-draft] — 2026-04-17

- First compiled `main.pdf` with five phase portraits (one per seed).
- Scaffolded Experiments, Discussion, Conclusion sections.

## [track3-v0.1] — 2026-04-17

- `track3_deploy/` — pure-NumPy neural surrogate (`NeuralSurrogate`),
  MiniLM query encoder (`QueryEncoder`) with stub fallback, streaming
  runner (`StreamingRunner`), JAX-based training (`SurrogateTrainer`),
  latency microbench with p50 < 25 ms / p99 < 100 ms CI gates.
- Public API exposed at `kiki_flow_core.*`.

## [track2-v0.1] — 2026-04-17

- `track2_paper/` — `ParticleSimulator`, `FullJKOSolver`,
  `T2FreeEnergy` (4-term functional), `MultiscaleLoop`, 5 figure
  generators, `paper_run` driver.
- PCA projection helper for Aeon embeddings.

## [track1-v0.1] — 2026-04-17

- `track1_perf/` — `T1FreeEnergy` (phenomenological 3-term),
  `EulerianGridSolver`, `DriftTrigger`, `offline_consolidator` CLI,
  checkpoint I/O via safetensors.

## Earlier (unreleased) commits — 2026-04-17

- Bootstrap: `pyproject.toml`, pre-commit hooks, CI workflow,
  pytest + coverage + ruff + mypy strict.
- `kiki_flow_core/` shared core: `FlowState` (Pydantic v2),
  `JKOStep`, `wasserstein_ops` (prox_w2, w2_distance, Sinkhorn),
  `CanonicalSpecies` (Levelt-Baddeley coupling from YAML),
  `MixedCanonicalSpecies` (4 × N projection), `AdvectionDiffusion`,
  `ScaffoldingScheduler`, `PhonologicalLoop`, `hooks` (Aeon, MoE,
  Routing with circuit breaker), `telemetry` (JSON logger +
  Prometheus metrics), end-to-end smoke + double-run determinism.
- 56 core tests passing, 95 % coverage.
