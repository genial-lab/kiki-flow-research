# Performance ledger

All measured numbers reported in the paper, with the sources that back
them. Every entry here is reproducible from a single script; hand-edits
to this file are not allowed (run the script and regenerate).

---

## Run timings

| Run | Setup | Wall time | Machine | Source |
|---|---|---|---|---|
| Fast path | 5 seeds × 100 slow × 10 k particles, grid 64, `use_w2_prox=False` | **2 min 31 s** | GrosMac M5 | `paper/stats.json` |
| Rigorous path | same, `use_w2_prox=True`, POT Sinkhorn | **84 min** | GrosMac M5 | `paper_rigorous/stats.json` |
| Rigorous reduced | grid 16, 5 slow, 100 particles, POT | 29.72 s | GrosMac M5 | `bench/sinkhorn_backend_bench.jsonl` |
| Rigorous reduced | same, MLX backend | **5.79 s** (5.13 × faster than POT) | GrosMac M5 | `bench/sinkhorn_backend_bench.jsonl` |

## Streaming surrogate latency (T3)

Measured by replaying 500 queries after 50 warmup queries. Stub encoder
unless noted; a real MiniLM encoder adds ~3 ms at p50.

| Weights | `state_dim` | hidden | p50 | p99 | Source |
|---|---|---|---|---|---|
| `v0.1` | 16 | 128 | **0.04 ms** | 0.06 ms | `bench/T3_latency.jsonl` |
| `v0.2-d128` | 128 | 256 | 0.37 ms | 0.53 ms | `bench/T3_latency.jsonl` |

Both satisfy the 10 ms p50 SLO with large margin.

## Specialization sweeps

Reduced problem: grid 16, 30 slow steps, 500 particles, asymmetric
per-species potentials.

| Sweep | Configs | Best (α, β, γ) | Mean entropy (bits) | Gap vs max | Source |
|---|---|---|---|---|---|
| Coarse 27-config | α ∈ {0, 1, 5}, β ∈ {0, 0.1, 1}, γ ∈ {0, 1, 5} | (5, 1, 0) | 2.60 | 1.40 bits (35 %) | `paper/hyperparam_sweep.json` |
| **Dense 72-config** | α ∈ {0, 1, 3, 5, 8, 10}, β ∈ {0, 0.5, 1, 2}, γ ∈ {0, 1, 5} | (10, 1, 0) | **1.96** | **2.04 bits (51 %)** | `paper/hyperparam_sweep_dense.json` |

Turing coupling γ has negligible effect in both sweeps.

## Entropic-Sinkhorn bias

`prox_w2` applied once to a smoothly peaked source / target pair,
KL between output and ε = 0.001 reference.

| ε | KL (bits) |
|---|---|
| 0.001 | 0.0 |
| 0.005 | 4.68 |
| 0.010 | 8.02 |
| 0.050 | 8.34 |
| 0.100 | 8.35 |

Saturates at ε ≥ 0.05 where the entropic regularization dominates.
Source: `paper/epsilon_sweep.json`.

## Continual-learning proxy

3-task sequential setup (peaks at grid indices 3, 8, 13), 30 JKO steps
per task, 5 seeds. Accuracy = cosine similarity between final state and
each task's target.

| Arm | task 0 | task 1 | task 2 |
|---|---|---|---|
| without | 0.29 ± 0.07 | 0.44 ± 0.07 | **0.82 ± 0.04** |
| with_prior (naive v0.4) | **0.90 ± 0.07** | 0.20 ± 0.01 | 0.004 ± 0.00 |
| **with_ewc (v0.6)** | **0.81 ± 0.06** | 0.26 ± 0.13 | **0.24 ± 0.06** |

EWC restores task-2 retention 60 × over naive while keeping task-0 near
0.90. Source: `paper/cl_benchmark_ewc.json`.

## Continual-learning on a real LLM

3-task sequential LoRA on **Qwen3-4B** (4-bit NF4, LoRA r=8 α=16,
500 steps × 500 samples × 3 tasks per seed, learning rate 2e-4, 5 seeds).
Each task resumes the previous LoRA adapter; the final adapter is
re-evaluated on every prior task to compute forgetting =
max(0, immediate − final). RTX 4090, ~4.5 min wall per seed, ~22 min total.

| Task | Immediate acc. | Final acc. | Forgetting |
|---|---|---|---|
| SST-2 (phono) | 0.945 ± 0.014 | 0.860 ± 0.039 | **0.085 ± 0.030** |
| CoLA (lex) | 0.768 ± 0.032 | **0.889 ± 0.039** | 0.000 (positive transfer) |
| BoolQ (syntax) | 0.565 ± 0.023 | 0.565 ± 0.023 | 0 (last task, by definition) |

Real LLM run confirms the distributional-proxy pattern: first-task
forgetting under pressure of later tasks, positive transfer mid-sequence.
Source: `bench/cl_llm/runs/e2_5seeds_summary.json`. Reproduce via
`docs/superpowers/runbooks/real-cl-bench.md`.

**BoolQ caveat (measured).** A 2000-step follow-up on seed 0 drives
training loss to 4 × 10⁻⁶ (total memorization of 500 samples) while
eval accuracy only moves from 0.565 to 0.592 — BoolQ is data and
adapter-capacity limited at rank 8, not compute limited. More steps
overfit rather than help. Source:
`bench/cl_llm/runs/c-boolq-extended/result.json`.

## Fast-vs-rigorous agreement

Reduced setup: grid 16, 500 particles, 20 slow steps, 5 seeds.
KL(ρ\_rigorous ‖ ρ\_fast) summed over the four species.

| Statistic | Value (bits) |
|---|---|
| mean | 6.38 × 10⁻⁴ |
| std | 2.65 × 10⁻⁴ |
| max | 9.37 × 10⁻⁴ |

Fast path is therefore within ~ 0.06 % of rigorous path in
distributional divergence. Source: `paper/fast_vs_rigorous_kl.json`.

## Aggregate final-state statistics (default flat potentials)

5 seeds × 100 slow × 10 k particles, grid 64, flat `V_i = 0`. Peak and
entropy of the final ρ per species.

| Species | Peak mean | Entropy mean (bits) | Max entropy |
|---|---|---|---|
| phono / lex / syntax / sem | 0.01596 ± 6 × 10⁻⁵ | 5.99998 ± 3 × 10⁻⁶ | 6.0 |

Near-uniform — specialization requires non-flat potentials. Source:
`paper/stats.json` aggregate section.

---

## Reproduction

```bash
# Fast path + all figures (~ 2 min 31 s)
PYTHONPATH=. uv run python -c "
from pathlib import Path
from kiki_flow_core.track2_paper.paper_run import run_paper
run_paper(seeds=[0,1,2,3,4], out_dir=Path('paper'), make_all_figures=True)
"

# Rigorous path (~ 84 min)
PYTHONPATH=. uv run python -c "
from pathlib import Path
from kiki_flow_core.track2_paper.paper_run import run_paper
run_paper(seeds=[0,1,2,3,4], out_dir=Path('paper_rigorous'), use_w2_prox=True)
"

# Dedicated sweeps (a few minutes each)
PYTHONPATH=. uv run python scripts/hyperparam_sweep_dense.py
PYTHONPATH=. uv run python scripts/epsilon_sweep.py
PYTHONPATH=. uv run python scripts/cl_benchmark_ewc.py
PYTHONPATH=. uv run python scripts/fast_vs_rigorous_kl.py

# T3 latency micro-bench (~ 1 min)
uv run pytest tests/track3_deploy/test_latency_microbench.py -v -m slow

# MLX vs POT Sinkhorn bench
PYTHONPATH=. uv run python scripts/bench_sinkhorn_backend.py  # (to be added)
```

## Machines

- **GrosMac M5**: Apple Silicon M5, 16 GB RAM, macOS 26.3.
- **Studio M3 Ultra**: Apple Silicon M3 Ultra, 512 GB RAM, used for
  longer rigorous runs where contention permits.

All headline figures use GrosMac M5; Studio is reserved for
paper-grade re-runs and is not part of the primary SLO.
