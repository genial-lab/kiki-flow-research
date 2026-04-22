# kiki_flow_core — math core

Load-bearing library shared by all three tracks. Everything here is imported
from `kiki_flow_core/__init__.py`; adding a public symbol means updating
`__all__` and writing a test.

## Module map

- `state.py` — `FlowState` (Pydantic v2), `assert_invariants`. Single source
  of truth for simplex / mass / track-id constraints.
- `master_equation.py` — `FreeEnergy` ABC, `ZeroF`, `JKOStep` with
  strategy-pattern `prox_fn` (injectable MLX backend without circular import).
- `wasserstein_ops.py` — POT-backed `sinkhorn_cost`, `w2_distance`, `prox_w2`.
  Log-domain Sinkhorn only.
- `species/` — `SpeciesBase`, `CanonicalSpecies`, `MixedCanonicalSpecies`.
  Coupling coefficients live in `species/data/levelt_baddeley_coupling.yaml`.
- `modules/` — `AdvectionDiffusion`, `PhonologicalLoop`, `ScaffoldingScheduler`.
- `hooks/` — adapters for Aeon, MoE-LoRA, Routing. Advisory-only; must return
  `None` on failure (never raise into host process).
- `telemetry/` — `StructuredLogger`, `Metrics` (Prometheus).
- `track1_perf/`, `track2_paper/`, `track3_deploy/` — see root
  `CLAUDE.md` for the track split.

## Numerical-stability rules (domain-specific)

- Sinkhorn: always `method="sinkhorn_log"`. Epsilon below ~0.005 with the
  naive kernel collapses the transport plan silently.
- Densities: clip with `np.clip(rho, 1e-12, None)` before any `log(rho)`,
  `log(rho/prior)`, or division. 1e-12 matches the `neg_tol` in invariants.
- Renormalize after every perturbation on the simplex (see
  `FreeEnergy.grad_rho` — perturb, then `/ sum`, then evaluate).
- Gradient eps default is 1e-4. Override in subclasses with closed-form
  gradients when possible; numerical grad is O(rho.size) value evaluations.

## Adding a new `FreeEnergy`

1. Subclass, implement `value(state) -> float`.
2. Override `grad_rho` with an analytical gradient if you can — the base
   numerical version is a fallback.
3. `value` must be a real number on any valid `FlowState` (no NaN / Inf).
   If your functional diverges on degenerate densities, clip inside `value`,
   not in the caller.
4. Write a test pairing the analytical gradient with a finite-difference
   check on a Dirichlet sample.

## Track-local code

`track{1,2,3}_*` subpackages MUST NOT import from each other. If two tracks
need the same helper, move it up to this level and cover it with a test in
`tests/` root (not under `tests/track*/`).

## Anti-patterns (domain-specific)

- Returning a dict of densities without passing through `assert_invariants`.
- Using `np.random.seed(...)` in library code. Accept a seed, build a local
  `np.random.default_rng(seed)`, and thread it through — never touch global
  state from inside the core.
- `torch.*` anywhere. MLX for GPU math on Apple Silicon, JAX for anything
  that wants XLA, POT for reference OT.
- Swallowing Sinkhorn `UserWarning: did not converge`. Either fix the
  setup or raise it as an error in a wrapper; silent divergence corrupts
  downstream figures.
- Mutating `FlowState` in place. It's a Pydantic BaseModel — use
  `state.model_copy(update={...})` (see `JKOStep.step`).
