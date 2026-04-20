# kiki-flow-research

**A Wasserstein gradient flow engine for LLM consolidation, grounded in the
Levelt–Baddeley model of language production.**

Upstream numerical engine of the *dreamOfkiki* research program, part of
[Hypneum Lab](https://github.com/hypneum-lab). Author: Clément Saillant
(L'Electron Rare).

[![CI](https://github.com/hypneum-lab/kiki-flow-research/actions/workflows/ci.yml/badge.svg)](https://github.com/hypneum-lab/kiki-flow-research/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## What is this?

`kiki-flow-research` models the state of a large language model as four
psycholinguistic activation densities (phono / lex / syntax / sem) that
evolve under a Wasserstein-regularized gradient flow. The implementation
solves the JKO scheme numerically on commodity Apple Silicon, extracts
a compact neural surrogate for streaming inference, and integrates with
existing MoE-LoRA routers through an advisory-only callback.

See the [design spec](docs/superpowers/specs/2026-04-17-kiki-flow-core-design.md)
for the full formalism and the [draft paper](paper/main.pdf) for the
quantitative results over five seeds.

## Three parallel tracks

| Track | Purpose | Performance | Status |
|---|---|---|---|
| `track1_perf` | Nightly offline consolidation over the LoRA stack tensor (40 hybrid species). | ~30 min on Studio M3 Ultra. | tag `track1-v0.1` |
| `track2_paper` | N-particle Langevin × JKO rigorous proxy for paper figures (4 Levelt-Baddeley species). | 2 min 31 s fast / 84 min rigorous on GrosMac M5. | tag `track2-v0.1` |
| `track3_deploy` | Pure-NumPy streaming surrogate, MiniLM query encoder, advisory-only routing. | p50 = 0.04 ms on GrosMac M5. | tag `track3-v0.1` |

All three tracks share `kiki_flow_core/` (state, master equation,
Wasserstein ops, species, modules, hooks, telemetry). 93 tests, 95 %
coverage, strict ruff + mypy.

## Install

```bash
git clone https://github.com/hypneum-lab/kiki-flow-research.git
cd kiki-flow-research
uv sync --all-extras
uv run pre-commit install
```

Tested on macOS 26.3 (arm64) with Python 3.14, MLX 0.31, JAX 0.10,
POT 0.9. Other platforms may work but have not been validated.

## Reproduce the paper

The paper's five figures are produced by three scripts. Each is
deterministic under the given seeds.

```bash
# Figure 1-3: phase portraits + F-decay + Turing profiles (5 seeds, fast path, ~2 min 31 s)
PYTHONPATH=. uv run python -c "
from pathlib import Path
from kiki_flow_core.track2_paper.paper_run import run_paper
run_paper(seeds=[0,1,2,3,4], out_dir=Path('paper'), make_all_figures=True)
"

# Figure 4: measured Sinkhorn epsilon bias
PYTHONPATH=. uv run python scripts/epsilon_sweep.py

# Figure 5: measured continual-learning proxy (5 seeds)
PYTHONPATH=. uv run python scripts/cl_benchmark.py

# Hyperparameter sweep (72 configs, dense) to reproduce the specialization finding
PYTHONPATH=. uv run python scripts/hyperparam_sweep.py

# Compile the LaTeX paper (requires tectonic or a TeX distribution)
cd paper && tectonic main.tex
```

### Real-LLM CL benchmark (§3.4, figure 7)

The §3.4 table and fig7 come from a real LoRA fine-tuning run on kxkm-ai
(RTX 4090). The orchestrator has three modes:

```bash
# Stub — deterministic summary, runs in CI (no network, no GPU)
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode stub --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/stub_0 --seed 0

# Preflight — read-only SSH probe of the remote host
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode preflight --ssh-host kxkm-ai \
  --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/preflight_check --seed 0

# Real — gated by --i-confirm-heavy-training; see runbook for prereqs
PYTHONPATH=. uv run python -m scripts.cl_llm_bench.run_cl_bench \
  --mode real --i-confirm-heavy-training --ssh-host kxkm-ai \
  --tasks phono_sst2,lex_cola,syn_boolq \
  --output bench/cl_llm/runs/my_run --seed 0 \
  --max-samples 5000 --n-steps 1500 --base-model Qwen/Qwen3-4B
```

Full prerequisites, GPU-time budget, and per-seed sweep recipe are in
[`docs/superpowers/runbooks/real-cl-bench.md`](docs/superpowers/runbooks/real-cl-bench.md).

Run the full test suite:

```bash
uv run pytest --cov=kiki_flow_core
```

## Quantitative results (paper v0.4)

| Claim | Measured value | Source |
|---|---|---|
| Flow tractable at scale | 2 min 31 s fast / 84 min rigorous on 5 seeds × 100 slow × 10k particles | `paper/stats.json`, `paper_rigorous/stats.json` |
| Emergent specialization under asymmetric potentials | Mean entropy 2.60 bits vs max 4.00 bits (gap 1.40 bits, 35 % below max) at `(α, β, γ) = (5, 1, 0)` | `paper/hyperparam_sweep.json` |
| Streaming surrogate latency | p50 = 0.04 ms, p99 = 0.06 ms (v0.1, `state_dim=16`); p50 = 0.37 ms (v0.2-d128, `state_dim=128`) | `bench/T3_latency.jsonl` |
| Continual-learning retention (honest trade-off) | Task 1 with cons. 0.90 ± 0.07 / without 0.29 ± 0.07; Task 3 with 0.004 / without 0.81 | `paper/cl_benchmark.json` |
| Real-LLM continual learning, 500-sample regime (Qwen3-4B 4-bit LoRA, 5 seeds) | SST-2: forgetting 0.085 ± 0.030; CoLA: apparent positive transfer; BoolQ: 0.565 ± 0.023 (data-limited) | `bench/cl_llm/runs/e2_5seeds_summary.json` |
| Real-LLM continual learning, 5000-sample regime (5 seeds) | SST-2: forgetting **0.361 ± 0.010** (4.2× higher); CoLA: forgetting 0.129 ± 0.021; positive-transfer artefact disappears | `bench/cl_llm/runs/e3_5k_5seeds_summary.json` |
| Sinkhorn entropic bias | KL(ρ_ε ‖ ρ_0.001) grows 0 → 8.3 bits, saturates at ε ≥ 0.05 | `paper/epsilon_sweep.json` |
| MLX Sinkhorn speedup vs POT | 5.13× faster on T2 JKO solver (GrosMac M5, `sinkhorn_backend="mlx"`) | `bench/T2_backend_speedup.jsonl` |

Results that did **not** hold are reported honestly in the paper:
the default flat-potential setup does not induce specialization
(entropy ≈ maximum), and naive prior-averaged consolidation
over-stabilizes on the first task at the expense of later ones.

## Integration with micro-kiki

This repository ships three artifacts to bridge to an existing
micro-kiki deployment:

1. `kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors` — trained
   surrogate weights matched to a 32-stack × 4-ortho = 128-dim state.
2. Two unified-diff patches and a design document in
   `docs/superpowers/patches/` that resolve three of the four known
   integration dependencies.
3. A working `KikiFlowBridge` reference implementation (in
   `patches/02-micro-kiki-dep3-runner-factory.md`) that owns its own
   tokenizer and returns None on any failure, guaranteeing zero impact
   on the nominal routing pathway when `KIKI_FLOW_ENABLED=0`.

See [`docs/superpowers/integration-notes.md`](docs/superpowers/integration-notes.md)
for the full status of each dependency.

## Repository layout

```
kiki-flow-research/
├── kiki_flow_core/                 # 45 source files, strict mypy + ruff
│   ├── state.py                    # FlowState Pydantic v2 + invariants
│   ├── master_equation.py          # FreeEnergy ABC + JKOStep
│   ├── wasserstein_ops.py          # prox_w2, w2_distance, Sinkhorn
│   ├── species/                    # OrthoSpecies, HybridSpecies
│   ├── modules/                    # AdvectionDiffusion, scheduler, phono loop
│   ├── hooks/                      # Aeon, MoE-LoRA, Routing adapters
│   ├── telemetry/                  # StructuredLogger, Metrics
│   ├── track1_perf/                # Eulerian grid + offline consolidator
│   ├── track2_paper/               # Particles + JKO + figures
│   └── track3_deploy/              # Pure-numpy surrogate + streaming runner
├── tests/                          # 93 tests, 95 % coverage
├── scripts/                        # Reproducibility scripts
│   └── cl_llm_bench/               # Real-LLM CL: stub / preflight / real modes
│       └── kxkm_trainer/           # Standalone 4-bit QLoRA trainer (runs on kxkm-ai)
├── paper/                          # main.tex, main.pdf, figures, JSON stats
├── paper_rigorous/                 # 84-min rigorous run artifacts
├── bench/                          # SLO latency records
│   └── cl_llm/runs/                # Per-seed real-CL manifests + summary
└── docs/superpowers/               # Specs, plans, integration, patches
    └── runbooks/real-cl-bench.md   # Real-mode CL manual runbook
```

## Citation

Draft pre-print, tagged `paper-v0.4-draft`:

```bibtex
@unpublished{kiki-flow-2026,
  author = {Saillant, Clément},
  title  = {Hybrid Cognitive-Fluid Dynamics for Language Production:
            A Wasserstein Gradient Flow Perspective},
  year   = 2026,
  url    = {https://github.com/hypneum-lab/kiki-flow-research},
  note   = {Draft; code, data, and figures released under MIT at tag paper-v0.4-draft}
}
```

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgments

Built on top of the excellent open-source work of
[Apple MLX](https://github.com/ml-explore/mlx),
[POT (Python Optimal Transport)](https://github.com/PythonOT/POT),
[JAX](https://github.com/google/jax),
[scikit-learn](https://github.com/scikit-learn/scikit-learn), and
[sentence-transformers](https://github.com/UKPLab/sentence-transformers).
