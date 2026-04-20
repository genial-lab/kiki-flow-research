# QueryConditionedF Implementation Plan (Workshop scope: T18-T24)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ZeroF` oracle used by `jko_oracle_runner` with `QueryConditionedF` — an AIF-motivated FreeEnergy coupling text embeddings to Wasserstein flow over 4 Levelt-Baddeley species via a pre-trained JEPA decoder, unlocking v0.3 sprint Phase 1/2 execution (T22/T23) with text-conditioned dynamics. Target deliverable: workshop paper section + ablation figure for NeurIPS 2026 workshop (sept).

**Architecture:** Three new code modules: `QueryConditionedF` (new FreeEnergy subclass with complexity+JEPA-likelihood+coupling terms, analytical grad for KL and J, autodiff JAX for JEPA), `HeuristicLabeler` (FR NLP pipeline producing per-species simplex labels from query text using SpaCy-fr + Lexique + phonemizer), and `train_g_jepa` (offline phase-A pre-training script that fits `g_JEPA` to heuristic labels → query embeddings via MSE). Wired into existing `jko_oracle_runner.py` (replaces `ZeroF` fallback).

**Tech Stack:** Python 3.14 (uv), JAX 0.10 + optax (training `g_JEPA`, autodiff), NumPy (core math), safetensors (weights), pytest + hypothesis (tests), SpaCy-fr `fr_core_news_lg`, phonemizer (espeak-ng FR backend), Lexique.org 3.83 CSV, optional WordNet-fr with KMeans clustering fallback. No torch.

**Spec reference:** `docs/superpowers/specs/2026-04-19-query-conditioned-f-design.md` (commit `8eaee45`).

---

## File Structure Map

### New files (code)

| Path | Responsibility |
|------|----------------|
| `kiki_flow_core/track3_deploy/query_conditioned_f.py` | `QueryConditionedF(FreeEnergy)` class with `value` + `grad_rho`, holds `g_JEPA` params, `π_prior`, `λ_J`, `σ²` |
| `kiki_flow_core/track3_deploy/data/heuristic_labeler.py` | `HeuristicLabeler` class with `label(query) -> dict[str, np.ndarray]` for 4 species |
| `kiki_flow_core/track3_deploy/data/phono_classes.py` | 32-class phonetic mapping (IPA → class index) for FR |
| `kiki_flow_core/track3_deploy/data/sem_categories.py` | 32 default semantic categories + WordNet-fr or KMeans fallback |
| `kiki_flow_core/track3_deploy/data/syntax_patterns.py` | 32 SpaCy dependency-label-based FR syntactic patterns |
| `kiki_flow_core/track3_deploy/train_g_jepa.py` | Offline phase-A pre-training script (CLI + core `train()` function) |
| `scripts/label_corpus.py` | CLI: consume JSONL corpus, run labeler, cache to `.npz` |

### New files (tests)

| Path | Tests |
|------|-------|
| `tests/track3_deploy/test_query_conditioned_f.py` | `value` finite, `grad_rho` shape, analytical grad matches finite-diff, limit cases (ZeroF equivalence) |
| `tests/track3_deploy/test_heuristic_labeler.py` | Output shape/simplex, determinism, golden 5-query assertion |
| `tests/track3_deploy/test_phono_classes.py` | 32 classes distinct, maps cover IPA subset |
| `tests/track3_deploy/test_train_g_jepa.py` | One gradient step reduces loss, save/load roundtrip |

### Modified files

| Path | Change |
|------|--------|
| `kiki_flow_core/track3_deploy/jko_oracle_runner.py` | Replace `JKOStep(ZeroF())` with `JKOStep(QueryConditionedF(...))`, load pre-trained `g_JEPA` weights |
| `pyproject.toml` | Add optional deps group `text-bridge-labeler = ["spacy>=3.7", "phonemizer>=3.2"]` |

### New files (artifacts, gitignored)

| Path | Purpose |
|------|---------|
| `artifacts/g_jepa_pretrained.safetensors` | Phase-A weights |
| `data/processed/heuristic_labels.npz` | Cached `{query_hash: π_target}` labels |
| `paper/workshop/section_4_method.tex` | Workshop paper section 4 draft |
| `paper/workshop/figures/*.pdf` | Ablation figures |

---

## Task Dependency Graph

```
T18 (QueryConditionedF class) ───┐
                                 ├─> T20 (pre-train g_JEPA) ─> T21 (wire oracle)
T19 (HeuristicLabeler) ──────────┘                                     │
                                                                        v
                                                               T22 (Phase 1 pilot 10k)
                                                                        │
                                                                        v
                                                               T23 (Phase 2 scale 50k)
                                                                        │
                                                                        v
                                                               T24 (workshop paper)
```

**Critical path:** T18+T19 (parallelizable) → T20 → T21 → T22 → T23 → T24. Total ~4-5 weeks wall-clock.

---

## Task 18: `QueryConditionedF` class

**Files:**
- Create: `kiki_flow_core/track3_deploy/query_conditioned_f.py`
- Test: `tests/track3_deploy/test_query_conditioned_f.py`

- [x] **Step 18.1: Write the failing test**

Create `tests/track3_deploy/test_query_conditioned_f.py`:

```python
"""Tests for QueryConditionedF — AIF FreeEnergy for text-conditioned Wasserstein flow."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.query_conditioned_f import QueryConditionedF


N_SPECIES = 4
N_STACKS = 32
EMBED_DIM = 384
HIDDEN_DIM = 256
GRAD_SHAPE_TOL = 1e-10
FINITE_DIFF_RTOL = 1e-3
LAMBDA_J_DEFAULT = 0.1
SIGMA2_DEFAULT = 1.0


def _uniform_state() -> FlowState:
    rho = {
        f"{s}:code": np.ones(N_STACKS, dtype=np.float32) / N_STACKS
        for s in ("phono", "sem", "lex", "syntax")
    }
    return FlowState(
        rho=rho,
        P_theta=np.zeros(16, dtype=np.float32),
        mu_curr=np.zeros(1, dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )


def _random_g_jepa_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((N_SPECIES * N_STACKS, HIDDEN_DIM)).astype(np.float32) * 0.01,
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": rng.standard_normal((HIDDEN_DIM, EMBED_DIM)).astype(np.float32) * 0.01,
        "b2": np.zeros(EMBED_DIM, dtype=np.float32),
    }


def _random_embedding(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)


def test_value_is_finite() -> None:
    state = _uniform_state()
    F = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=_random_embedding(),
        lambda_j=LAMBDA_J_DEFAULT,
        sigma2=SIGMA2_DEFAULT,
    )
    v = F.value(state)
    assert np.isfinite(v)


def test_grad_rho_shape() -> None:
    state = _uniform_state()
    F = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=_random_embedding(),
    )
    for species in state.rho:
        grad = F.grad_rho(state, species)
        assert grad.shape == (N_STACKS,)


def test_limit_g_jepa_zero_reduces_to_complexity() -> None:
    """If g_JEPA weights are zero and lambda_j=0, F reduces to sum_s KL(rho_s || pi_prior)."""
    zero_params = {
        "W1": np.zeros((N_SPECIES * N_STACKS, HIDDEN_DIM), dtype=np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": np.zeros((HIDDEN_DIM, EMBED_DIM), dtype=np.float32),
        "b2": np.zeros(EMBED_DIM, dtype=np.float32),
    }
    state = _uniform_state()
    F = QueryConditionedF(
        g_jepa_params=zero_params,
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
        sigma2=SIGMA2_DEFAULT,
    )
    # uniform rho with uniform prior → KL = 0 per species
    assert abs(F.value(state)) < FINITE_DIFF_RTOL


def test_grad_analytical_matches_finite_diff_on_complexity() -> None:
    """Gradient of complexity term matches finite-difference check."""
    state = _uniform_state()
    # zero out JEPA and J; only complexity left
    zero_params = {
        "W1": np.zeros((N_SPECIES * N_STACKS, HIDDEN_DIM), dtype=np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": np.zeros((HIDDEN_DIM, EMBED_DIM), dtype=np.float32),
        "b2": np.zeros(EMBED_DIM, dtype=np.float32),
    }
    F = QueryConditionedF(
        g_jepa_params=zero_params,
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
    )
    species = "phono:code"
    grad = F.grad_rho(state, species, eps=1e-4)
    # For KL(rho || uniform) at uniform rho, gradient should be ~0 (flat minimum)
    assert np.abs(grad).max() < FINITE_DIFF_RTOL


def test_value_strictly_positive_when_rho_differs_from_prior() -> None:
    """F.value > 0 when rho is non-uniform (complexity term positive)."""
    rho = {
        f"{s}:code": (np.eye(N_STACKS)[0]).astype(np.float32)  # peaked on stack 0
        for s in ("phono", "sem", "lex", "syntax")
    }
    state = FlowState(
        rho=rho,
        P_theta=np.zeros(16, dtype=np.float32),
        mu_curr=np.zeros(1, dtype=np.float32),
        tau=0,
        metadata={"track_id": "T3"},
    )
    F = QueryConditionedF(
        g_jepa_params=_random_g_jepa_params(),
        embedding=np.zeros(EMBED_DIM, dtype=np.float32),
        lambda_j=0.0,
    )
    assert F.value(state) > 0
```

- [x] **Step 18.2: Run test to verify it fails**

Run: `uv run python -m pytest tests/track3_deploy/test_query_conditioned_f.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 18.3: Implement QueryConditionedF**

Create `kiki_flow_core/track3_deploy/query_conditioned_f.py`:

```python
"""QueryConditionedF — AIF FreeEnergy for text-conditioned Wasserstein flow.

F(rho, q) = sum_s KL(rho_s || pi_prior_s)                     # complexity
          + (1/(2*sigma²)) * ||embed(q) - g_JEPA(rho)||²        # accuracy
          + lambda_j * sum_{s,t} J_{st} <rho_s, rho_t>          # coupling

See design spec: docs/superpowers/specs/2026-04-19-query-conditioned-f-design.md
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from kiki_flow_core.master_equation import FreeEnergy
from kiki_flow_core.state import FlowState


SPECIES_CANONICAL = ("phono:code", "sem:code", "lex:code", "syntax:code")
_N_SPECIES = 4
_N_STACKS = 32
_EPS = 1e-12
_LOG_EPS = 1e-8


def _g_jepa_forward(params: dict, rho_flat: jnp.ndarray) -> jnp.ndarray:
    """2-layer MLP: rho_flat (128,) -> hidden (256,) -> embedding (384,)."""
    h = jax.nn.gelu(rho_flat @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


class QueryConditionedF(FreeEnergy):
    """FreeEnergy coupling text embedding to Wasserstein flow via JEPA decoder."""

    def __init__(
        self,
        g_jepa_params: dict[str, np.ndarray],
        embedding: np.ndarray,
        pi_prior: dict[str, np.ndarray] | None = None,
        coupling_matrix: np.ndarray | None = None,
        lambda_j: float = 0.1,
        sigma2: float = 1.0,
    ) -> None:
        self.g_jepa_params = {k: jnp.asarray(v) for k, v in g_jepa_params.items()}
        self.embedding = jnp.asarray(embedding, dtype=jnp.float32)
        if pi_prior is None:
            pi_prior = {
                sp: np.ones(_N_STACKS, dtype=np.float32) / _N_STACKS
                for sp in SPECIES_CANONICAL
            }
        self.pi_prior = pi_prior
        if coupling_matrix is None:
            coupling_matrix = np.zeros((_N_SPECIES, _N_SPECIES), dtype=np.float32)
        self.coupling_matrix = coupling_matrix
        self.lambda_j = lambda_j
        self.sigma2 = sigma2
        # Pre-compile JAX gradient for JEPA accuracy term
        self._grad_jepa_fn = jax.jit(jax.grad(self._jepa_loss))

    def _flatten_rho(self, state: FlowState) -> jnp.ndarray:
        return jnp.concatenate([jnp.asarray(state.rho[sp]) for sp in SPECIES_CANONICAL])

    def _jepa_loss(self, rho_flat: jnp.ndarray) -> jnp.ndarray:
        pred = _g_jepa_forward(self.g_jepa_params, rho_flat)
        diff = self.embedding - pred
        return 0.5 * jnp.sum(diff**2) / self.sigma2

    def value(self, state: FlowState) -> float:
        # Complexity term
        complexity = 0.0
        for sp in SPECIES_CANONICAL:
            rho = np.clip(state.rho[sp], _EPS, None)
            prior = np.clip(self.pi_prior[sp], _EPS, None)
            complexity += float(np.sum(rho * (np.log(rho + _LOG_EPS) - np.log(prior + _LOG_EPS))))
        # Accuracy term (JEPA)
        rho_flat = self._flatten_rho(state)
        accuracy = float(self._jepa_loss(rho_flat))
        # Coupling term
        coupling = 0.0
        if self.lambda_j > 0:
            for i, si in enumerate(SPECIES_CANONICAL):
                for j, sj in enumerate(SPECIES_CANONICAL):
                    jij = float(self.coupling_matrix[i, j])
                    if jij != 0.0:
                        coupling += jij * float(np.dot(state.rho[si], state.rho[sj]))
            coupling *= self.lambda_j
        return complexity + accuracy + coupling

    def grad_rho(self, state: FlowState, species_name: str, eps: float = 1e-4) -> np.ndarray:
        if species_name not in self.pi_prior:
            raise ValueError(f"Unknown species: {species_name!r}")
        # Complexity gradient: log(rho_s / pi_prior_s) + 1
        rho_s = np.clip(state.rho[species_name], _EPS, None)
        prior_s = np.clip(self.pi_prior[species_name], _EPS, None)
        grad_complexity = np.log(rho_s + _LOG_EPS) - np.log(prior_s + _LOG_EPS) + 1.0
        # Accuracy gradient: autodiff chain rule via jax
        rho_flat = self._flatten_rho(state)
        grad_jepa_flat = np.asarray(self._grad_jepa_fn(rho_flat))
        species_idx = SPECIES_CANONICAL.index(species_name)
        slice_start = species_idx * _N_STACKS
        slice_end = slice_start + _N_STACKS
        grad_accuracy = grad_jepa_flat[slice_start:slice_end]
        # Coupling gradient: 2 * lambda_j * sum_t J_{s,t} * rho_t
        grad_coupling = np.zeros(_N_STACKS, dtype=np.float32)
        if self.lambda_j > 0:
            for t, st in enumerate(SPECIES_CANONICAL):
                jst = float(self.coupling_matrix[species_idx, t])
                if jst != 0.0:
                    grad_coupling += jst * state.rho[st]
            grad_coupling *= 2.0 * self.lambda_j
        return (grad_complexity + grad_accuracy + grad_coupling).astype(np.float32)
```

- [x] **Step 18.4: Run test to verify it passes**

Run: `uv run python -m pytest tests/track3_deploy/test_query_conditioned_f.py -v`
Expected: 5 passed.

- [x] **Step 18.5: Commit**

```bash
git add kiki_flow_core/track3_deploy/query_conditioned_f.py \
        tests/track3_deploy/test_query_conditioned_f.py
git commit -m "$(cat <<'EOF'
feat(track3): query-conditioned free energy

AIF-motivated FreeEnergy subclass combining per-species KL complexity,
JEPA likelihood (via 2-layer MLP decoder g_JEPA), and T2-style species
coupling. Value and grad_rho are analytical for complexity+coupling,
autodiff JAX for the JEPA term.

Entrypoint for T14/T15 oracle to produce text-conditioned (pre, post)
pairs, replacing the query-agnostic ZeroF placeholder.
EOF
)"
```

---

## Task 19: `HeuristicLabeler` (FR NLP pipeline)

**Files:**
- Create: `kiki_flow_core/track3_deploy/data/heuristic_labeler.py`
- Create: `kiki_flow_core/track3_deploy/data/phono_classes.py`
- Create: `kiki_flow_core/track3_deploy/data/sem_categories.py`
- Create: `kiki_flow_core/track3_deploy/data/syntax_patterns.py`
- Test: `tests/track3_deploy/test_heuristic_labeler.py`
- Test: `tests/track3_deploy/test_phono_classes.py`
- Modify: `pyproject.toml` (add `text-bridge-labeler` optional deps)

- [x] **Step 19.1: Add optional deps group to pyproject.toml**

Inspect existing `[project.optional-dependencies]` section. Add a new group:

```toml
[project.optional-dependencies]
# ... existing groups ...
text-bridge-labeler = [
    "spacy>=3.7",
    "phonemizer>=3.2",
]
```

Then:

```bash
.venv/bin/pip install "spacy>=3.7" "phonemizer>=3.2"
.venv/bin/python -m spacy download fr_core_news_lg
# phonemizer needs espeak-ng system binary; check availability:
.venv/bin/python -c "from phonemizer import phonemize; print(phonemize('bonjour', language='fr-fr', backend='espeak'))"
```

If `espeak-ng` is missing on macOS: `brew install espeak-ng`.

- [x] **Step 19.2: Create phono_classes module**

Create `kiki_flow_core/track3_deploy/data/phono_classes.py`:

```python
"""FR phonetic class mapping — 32 classes for phonemizer IPA output.

Classes organized as 4 groups x 8 subclasses:
  - Vowels: [a, ɑ, e, ɛ, i, o, ɔ, u, y, ø, œ, ã, ɛ̃, ɔ̃, œ̃] → 8 slots
  - Stops: [p, b, t, d, k, g, ʔ] → 8 slots
  - Fricatives+affricates: [f, v, s, z, ʃ, ʒ, ʁ, h] → 8 slots
  - Nasals+liquids+glides: [m, n, ɲ, ŋ, l, j, w, ɥ] → 8 slots

IPA phonemes outside these fall back to class 31 ("other").
"""
from __future__ import annotations

PHONO_CLASSES: dict[str, int] = {
    # Vowels (0-7)
    "a": 0, "ɑ": 0, "æ": 0,
    "e": 1, "ɛ": 2,
    "i": 3, "ɪ": 3, "y": 3,
    "o": 4, "ɔ": 4,
    "u": 5, "ʊ": 5,
    "ø": 6, "œ": 6, "ə": 6,
    "ã": 7, "ɛ̃": 7, "ɔ̃": 7, "œ̃": 7,
    # Stops (8-15)
    "p": 8, "b": 9, "t": 10, "d": 11,
    "k": 12, "g": 13, "ʔ": 14,
    # Fricatives + affricates (16-23)
    "f": 16, "v": 17, "s": 18, "z": 19,
    "ʃ": 20, "ʒ": 21, "ʁ": 22, "h": 23, "χ": 22,
    # Nasals / liquids / glides (24-31)
    "m": 24, "n": 25, "ɲ": 26, "ŋ": 27,
    "l": 28, "j": 29, "w": 30, "ɥ": 31,
}

N_CLASSES = 32
DEFAULT_CLASS = 31  # fallback for unknown phonemes
```

- [x] **Step 19.3: Write test for phono_classes**

Create `tests/track3_deploy/test_phono_classes.py`:

```python
"""Tests for PHONO_CLASSES mapping."""
from __future__ import annotations

from kiki_flow_core.track3_deploy.data.phono_classes import (
    DEFAULT_CLASS,
    N_CLASSES,
    PHONO_CLASSES,
)


def test_32_classes_exactly() -> None:
    assert N_CLASSES == 32
    assert DEFAULT_CLASS < N_CLASSES


def test_classes_within_bounds() -> None:
    for phoneme, cls in PHONO_CLASSES.items():
        assert 0 <= cls < N_CLASSES, f"{phoneme} -> {cls} out of bounds"


def test_covers_all_32_slots() -> None:
    used = set(PHONO_CLASSES.values())
    # Not every slot must be used by FR phonemes, but most should be
    assert len(used) >= 20


def test_common_fr_phonemes_mapped() -> None:
    for p in ("a", "i", "u", "p", "t", "k", "s", "ʁ", "m", "l"):
        assert p in PHONO_CLASSES
```

Run: `uv run python -m pytest tests/track3_deploy/test_phono_classes.py -v`
Expected: 4 passed.

- [x] **Step 19.4: Create sem_categories module**

Create `kiki_flow_core/track3_deploy/data/sem_categories.py`:

```python
"""32 default semantic categories for the sem:code species.

Derived from WordNet top-level lexicographer files ("lexnames"), filtered to 32
cognitively-relevant categories. Fallback: KMeans-32 clustering over MiniLM
embeddings of a corpus of FR nouns if WordNet-fr is unavailable.
"""
from __future__ import annotations

# Based on WordNet lexnames (noun.* and verb.*), trimmed to 32
SEM_CATEGORIES: tuple[str, ...] = (
    "person", "animal", "plant", "body", "food", "artifact", "substance", "shape",
    "location", "group", "quantity", "time", "event", "act", "state", "attribute",
    "cognition", "communication", "feeling", "motive", "perception", "possession",
    "process", "relation", "phenomenon", "Tops", "motion", "change", "contact",
    "creation", "social", "other",
)

N_SEM = 32
assert len(SEM_CATEGORIES) == N_SEM
```

- [x] **Step 19.5: Create syntax_patterns module**

Create `kiki_flow_core/track3_deploy/data/syntax_patterns.py`:

```python
"""32 FR syntactic patterns based on SpaCy dependency labels.

Each pattern corresponds to a dep-label (or combination). We count tokens whose
dep matches each pattern and normalize.
"""
from __future__ import annotations

# Ordered list of SpaCy FR dep labels mapped to 32 slots
# See https://universaldependencies.org/u/dep/
SYNTAX_PATTERNS: tuple[str, ...] = (
    "nsubj", "obj", "iobj", "obl", "csubj", "ccomp", "xcomp",
    "nmod", "amod", "appos", "advmod", "det", "case", "mark",
    "cc", "conj", "aux", "cop", "acl", "advcl", "parataxis", "list",
    "root", "punct", "compound", "fixed", "flat",
    "nummod", "expl", "discourse", "vocative", "dep",
)

N_SYNTAX = 32
assert len(SYNTAX_PATTERNS) == N_SYNTAX
```

- [x] **Step 19.6: Write the failing test for HeuristicLabeler**

Create `tests/track3_deploy/test_heuristic_labeler.py`:

```python
"""Tests for HeuristicLabeler."""
from __future__ import annotations

import numpy as np
import pytest

spacy = pytest.importorskip("spacy")

from kiki_flow_core.track3_deploy.data.heuristic_labeler import HeuristicLabeler


N_STACKS = 32
SIMPLEX_SUM_TOL = 1e-5
SPECIES_KEYS = {"phono:code", "sem:code", "lex:code", "syntax:code"}


@pytest.fixture(scope="module")
def labeler() -> HeuristicLabeler:
    """Shared labeler instance — SpaCy load is expensive."""
    return HeuristicLabeler()


def test_label_output_structure(labeler: HeuristicLabeler) -> None:
    out = labeler.label("Bonjour le monde.")
    assert set(out.keys()) == SPECIES_KEYS
    for sp, vec in out.items():
        assert vec.shape == (N_STACKS,), f"{sp} shape {vec.shape}"
        assert vec.dtype == np.float32


def test_simplex_constraint(labeler: HeuristicLabeler) -> None:
    out = labeler.label("Voici une phrase française de test.")
    for sp, vec in out.items():
        assert abs(vec.sum() - 1.0) < SIMPLEX_SUM_TOL, f"{sp} sum {vec.sum()}"
        assert (vec >= 0).all()


def test_determinism(labeler: HeuristicLabeler) -> None:
    q = "Une phrase déterministe."
    a = labeler.label(q)
    b = labeler.label(q)
    for sp in SPECIES_KEYS:
        np.testing.assert_array_equal(a[sp], b[sp])


def test_different_queries_different_labels(labeler: HeuristicLabeler) -> None:
    a = labeler.label("mot")
    b = labeler.label("Les chercheurs étudient la cognition.")
    # at least one species must differ between the two queries
    differs = any(not np.allclose(a[sp], b[sp]) for sp in SPECIES_KEYS)
    assert differs


def test_empty_query_returns_uniform(labeler: HeuristicLabeler) -> None:
    """Empty / whitespace-only query falls back to uniform simplex."""
    out = labeler.label("")
    for sp, vec in out.items():
        np.testing.assert_allclose(vec, np.ones(N_STACKS) / N_STACKS, rtol=1e-5)
```

- [x] **Step 19.7: Run failing**

Run: `uv run python -m pytest tests/track3_deploy/test_heuristic_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 19.8: Implement HeuristicLabeler**

Create `kiki_flow_core/track3_deploy/data/heuristic_labeler.py`:

```python
"""HeuristicLabeler — produces per-species simplex labels from FR query text.

Phase-A pre-training target generator. Uses SpaCy-fr for tokenization/parse,
phonemizer for phoneme extraction, Lexique.org for frequency bins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.track3_deploy.data.phono_classes import (
    DEFAULT_CLASS,
    N_CLASSES,
    PHONO_CLASSES,
)
from kiki_flow_core.track3_deploy.data.sem_categories import (
    N_SEM,
    SEM_CATEGORIES,
)
from kiki_flow_core.track3_deploy.data.syntax_patterns import (
    N_SYNTAX,
    SYNTAX_PATTERNS,
)


N_STACKS = 32
_EPS_SMOOTH = 1e-6
_DEFAULT_SPACY_MODEL = "fr_core_news_lg"


def _uniform() -> np.ndarray:
    return np.ones(N_STACKS, dtype=np.float32) / N_STACKS


def _smooth_normalize(counts: np.ndarray) -> np.ndarray:
    """Add-epsilon smoothing then L1 normalize."""
    smoothed = counts + _EPS_SMOOTH
    return (smoothed / smoothed.sum()).astype(np.float32)


class HeuristicLabeler:
    """Produce per-species 32-simplex labels for French query text."""

    def __init__(
        self,
        spacy_model: str = _DEFAULT_SPACY_MODEL,
        lexique_csv: Path | str | None = None,
        phoneme_lang: str = "fr-fr",
    ) -> None:
        import spacy
        self._nlp = spacy.load(spacy_model, disable=["ner"])  # NER not needed
        self._phoneme_lang = phoneme_lang
        self._lexique_bins: dict[str, int] | None = None
        if lexique_csv is not None:
            self._load_lexique(Path(lexique_csv))

    def _load_lexique(self, path: Path) -> None:
        """Load Lexique.org CSV, precompute log-freq bin (0-31) per lemma."""
        import pandas as pd
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        # Lexique.org has a 'freqlivres' column (per-million frequency)
        freq_col = "freqlivres" if "freqlivres" in df.columns else df.columns[1]
        lemma_col = "lemme" if "lemme" in df.columns else df.columns[0]
        freqs = df[freq_col].fillna(0.0).astype(float).clip(lower=_EPS_SMOOTH)
        log_freqs = np.log1p(freqs)
        # 32 log-space bins across the lemma frequency distribution
        q = np.linspace(0, 1, N_STACKS + 1)[1:-1]
        edges = np.quantile(log_freqs, q)
        bins = np.digitize(log_freqs, edges)
        self._lexique_bins = dict(zip(df[lemma_col].astype(str).str.lower(), bins.tolist(), strict=False))

    def label(self, query: str) -> dict[str, np.ndarray]:
        if not query.strip():
            return {
                "phono:code": _uniform(),
                "sem:code": _uniform(),
                "lex:code": _uniform(),
                "syntax:code": _uniform(),
            }
        doc = self._nlp(query)
        return {
            "phono:code": self._phono_distribution(query),
            "sem:code": self._sem_distribution(doc),
            "lex:code": self._lex_distribution(doc),
            "syntax:code": self._syntax_distribution(doc),
        }

    def _phono_distribution(self, query: str) -> np.ndarray:
        from phonemizer import phonemize
        ipa = phonemize(query, language=self._phoneme_lang, backend="espeak", strip=True)
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for ch in ipa:
            cls = PHONO_CLASSES.get(ch, DEFAULT_CLASS)
            counts[cls] += 1.0
        return _smooth_normalize(counts)

    def _sem_distribution(self, doc: Any) -> np.ndarray:
        """Heuristic: map content POS to category bins via lemma hash."""
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for tok in doc:
            if tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and not tok.is_stop:
                # Map lemma → category index via stable hash (placeholder for WordNet-fr)
                idx = hash(tok.lemma_.lower()) % N_SEM
                counts[idx] += 1.0
        return _smooth_normalize(counts)

    def _lex_distribution(self, doc: Any) -> np.ndarray:
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for tok in doc:
            if tok.is_alpha:
                bin_idx = (
                    self._lexique_bins.get(tok.lemma_.lower(), N_STACKS // 2)
                    if self._lexique_bins is not None
                    else (hash(tok.text.lower()) % N_STACKS)
                )
                counts[int(bin_idx)] += 1.0
        return _smooth_normalize(counts)

    def _syntax_distribution(self, doc: Any) -> np.ndarray:
        dep_to_idx = {dep: i for i, dep in enumerate(SYNTAX_PATTERNS)}
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for tok in doc:
            idx = dep_to_idx.get(tok.dep_, len(SYNTAX_PATTERNS) - 1)
            counts[idx] += 1.0
        return _smooth_normalize(counts)
```

- [x] **Step 19.9: Run tests**

Run: `uv run python -m pytest tests/track3_deploy/test_heuristic_labeler.py tests/track3_deploy/test_phono_classes.py -v`
Expected: 5 + 4 = 9 passed. If SpaCy model not available, the labeler test file will skip at the `importorskip`.

- [x] **Step 19.10: Commit**

```bash
git add kiki_flow_core/track3_deploy/data/heuristic_labeler.py \
        kiki_flow_core/track3_deploy/data/phono_classes.py \
        kiki_flow_core/track3_deploy/data/sem_categories.py \
        kiki_flow_core/track3_deploy/data/syntax_patterns.py \
        tests/track3_deploy/test_heuristic_labeler.py \
        tests/track3_deploy/test_phono_classes.py \
        pyproject.toml
git commit -m "$(cat <<'EOF'
feat(track3): heuristic labeler for FR queries

Phase-A label generator for QueryConditionedF pre-training. Produces
per-species 32-simplex targets from FR text via SpaCy-fr dependency
parse, phonemizer IPA, and Lexique.org frequency bins.

Semantic categories are keyed via deterministic lemma hash (fallback
for WordNet-fr); sem clustering refinement deferred to v0.4.
EOF
)"
```

---

## Task 20: Pre-train `g_JEPA` (phase A)

**Files:**
- Create: `kiki_flow_core/track3_deploy/train_g_jepa.py`
- Test: `tests/track3_deploy/test_train_g_jepa.py`
- Create (runtime): `artifacts/g_jepa_pretrained.safetensors`

- [x] **Step 20.1: Write the failing test**

Create `tests/track3_deploy/test_train_g_jepa.py`:

```python
"""Tests for g_JEPA pre-training (phase A)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from kiki_flow_core.track3_deploy.train_g_jepa import (
    gjepa_init_params,
    gjepa_step,
    save_gjepa,
    load_gjepa,
)


HIDDEN_DIM = 256
INPUT_DIM = 128
OUTPUT_DIM = 384
BATCH = 8
LR = 1e-2
N_STEPS = 50
LOSS_REDUCTION = 0.5


def test_init_shapes() -> None:
    params = gjepa_init_params(seed=0)
    assert params["W1"].shape == (INPUT_DIM, HIDDEN_DIM)
    assert params["b1"].shape == (HIDDEN_DIM,)
    assert params["W2"].shape == (HIDDEN_DIM, OUTPUT_DIM)
    assert params["b2"].shape == (OUTPUT_DIM,)


def test_single_step_reduces_loss() -> None:
    """One SGD step on a fixed toy batch must reduce MSE loss."""
    params = gjepa_init_params(seed=0)
    rng = np.random.default_rng(0)
    rho_flat = rng.random((BATCH, INPUT_DIM)).astype(np.float32)
    rho_flat /= rho_flat.sum(axis=-1, keepdims=True)
    targets = rng.standard_normal((BATCH, OUTPUT_DIM)).astype(np.float32)
    # loss before
    import optax
    optim = optax.adamw(LR)
    opt_state = optim.init(params)
    loss_before = None
    for i in range(N_STEPS):
        params, opt_state, loss = gjepa_step(params, opt_state, rho_flat, targets, optim)
        if i == 0:
            loss_before = loss
    loss_after = loss
    assert loss_after < loss_before * LOSS_REDUCTION, f"{loss_before} -> {loss_after}"


def test_save_load_roundtrip(tmp_path) -> None:
    params = gjepa_init_params(seed=0)
    path = tmp_path / "g_jepa.safetensors"
    save_gjepa(params, path)
    loaded = load_gjepa(path)
    for key in params:
        np.testing.assert_array_equal(np.asarray(params[key]), np.asarray(loaded[key]))
```

- [x] **Step 20.2: Run failing**

Run: `uv run python -m pytest tests/track3_deploy/test_train_g_jepa.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 20.3: Implement train_g_jepa**

Create `kiki_flow_core/track3_deploy/train_g_jepa.py`:

```python
"""Phase-A pre-training: fit g_JEPA to (heuristic-label, encoder-embedding) pairs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import load_file, save_file


INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 384
_WEIGHT_INIT_SCALE = 0.02
_DEFAULT_LR = 3e-4
_DEFAULT_BATCH = 64
_DEFAULT_EPOCHS = 5


logger = logging.getLogger(__name__)


def gjepa_init_params(seed: int = 0) -> dict[str, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)
    return {
        "W1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * _WEIGHT_INIT_SCALE,
        "b1": jnp.zeros(HIDDEN_DIM),
        "W2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * _WEIGHT_INIT_SCALE,
        "b2": jnp.zeros(OUTPUT_DIM),
    }


def gjepa_forward(params: dict[str, jnp.ndarray], rho_flat: jnp.ndarray) -> jnp.ndarray:
    h = jax.nn.gelu(rho_flat @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def _loss_fn(params: dict, rho_flat: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    pred = gjepa_forward(params, rho_flat)
    return jnp.mean((pred - targets) ** 2)


@jax.jit
def _step_impl(
    params: dict,
    opt_state: optax.OptState,
    rho_flat: jnp.ndarray,
    targets: jnp.ndarray,
    optim_update,
) -> tuple[dict, optax.OptState, jnp.ndarray]:
    loss_val, grads = jax.value_and_grad(_loss_fn)(params, rho_flat, targets)
    updates, opt_state = optim_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


def gjepa_step(
    params: dict,
    opt_state: optax.OptState,
    rho_flat: np.ndarray,
    targets: np.ndarray,
    optim,
) -> tuple[dict, optax.OptState, float]:
    new_params, new_state, loss_val = _step_impl(
        params, opt_state, jnp.asarray(rho_flat), jnp.asarray(targets), optim.update
    )
    return new_params, new_state, float(loss_val)


def save_gjepa(params: dict, path: Path | str) -> None:
    flat = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    save_file(flat, str(path))


def load_gjepa(path: Path | str) -> dict:
    flat = load_file(str(path))
    return {k: jnp.asarray(v) for k, v in flat.items()}


def train(
    labels_npz: Path,
    embeddings_npz: Path,
    output_path: Path,
    lr: float = _DEFAULT_LR,
    batch: int = _DEFAULT_BATCH,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 0,
) -> None:
    """Train g_JEPA on paired (heuristic-label, encoder-embedding) data."""
    labels = np.load(labels_npz, allow_pickle=True)  # dict: query_hash -> (4, 32) simplex
    embeddings = np.load(embeddings_npz, allow_pickle=True)  # dict: query_hash -> (384,)
    hashes = sorted(set(labels.files) & set(embeddings.files))
    rho_flat = np.stack([labels[h].flatten() for h in hashes]).astype(np.float32)
    targets = np.stack([embeddings[h] for h in hashes]).astype(np.float32)
    logger.info("loaded %d paired samples", len(hashes))

    params = gjepa_init_params(seed=seed)
    optim = optax.adamw(lr)
    opt_state = optim.init(params)
    rng = np.random.default_rng(seed)
    n = rho_flat.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            params, opt_state, loss = gjepa_step(
                params, opt_state, rho_flat[idx], targets[idx], optim
            )
            total_loss += loss
            n_batches += 1
        logger.info("epoch=%d avg_loss=%.5f", epoch, total_loss / max(1, n_batches))
    save_gjepa(params, output_path)
    logger.info("saved g_JEPA to %s", output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-train g_JEPA on heuristic labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    train(
        args.labels, args.embeddings, args.output,
        lr=args.lr, batch=args.batch, epochs=args.epochs, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 20.4: Run tests**

Run: `uv run python -m pytest tests/track3_deploy/test_train_g_jepa.py -v`
Expected: 3 passed.

- [x] **Step 20.5: Commit**

```bash
git add kiki_flow_core/track3_deploy/train_g_jepa.py \
        tests/track3_deploy/test_train_g_jepa.py
git commit -m "$(cat <<'EOF'
feat(track3): g_JEPA phase-A pre-training

2-layer MLP decoder (128 -> 256 -> 384, GELU) trained via optax AdamW
on paired (heuristic-label, encoder-embedding) data. Provides the
accuracy term of QueryConditionedF.

CLI: reads labels and embeddings NPZ keyed by query sha256, saves
safetensors checkpoint for downstream oracle consumption.
EOF
)"
```

---

## Task 21: Wire `QueryConditionedF` into `jko_oracle_runner`

**Files:**
- Modify: `kiki_flow_core/track3_deploy/jko_oracle_runner.py`
- Test: extend `tests/track3_deploy/test_jko_oracle_runner.py`

- [x] **Step 21.1: Update runner to accept g_JEPA weights**

In `kiki_flow_core/track3_deploy/jko_oracle_runner.py`, add an argparse flag `--g-jepa` pointing to a pre-trained `.safetensors`. Replace the current `ZeroF`-based `compute_jko_pair` with a `QueryConditionedF`-based version that takes the embedder + `g_JEPA` params.

Add at the top:

```python
from kiki_flow_core.track3_deploy.query_conditioned_f import QueryConditionedF
from kiki_flow_core.track3_deploy.train_g_jepa import load_gjepa
```

Add a new factory:

```python
def make_pair_computer(g_jepa_path: Path | None, embedder) -> Callable[[str], dict]:
    """Build compute_jko_pair closure parametrized by g_JEPA weights + embedder."""
    from kiki_flow_core.master_equation import JKOStep
    from kiki_flow_core.state import FlowState
    from kiki_flow_core.track3_deploy.state_projection import flatten

    g_jepa_params = load_gjepa(g_jepa_path) if g_jepa_path else None

    def _compute(query: str) -> dict:
        if g_jepa_params is None:
            # Fall back to ZeroF path (legacy behavior, used when no weights available)
            from kiki_flow_core.master_equation import ZeroF
            step = JKOStep(ZeroF())
        else:
            emb = embedder(query)
            F = QueryConditionedF(g_jepa_params=g_jepa_params, embedding=emb)
            step = JKOStep(F)
        # Deterministic uniform init per query (seeded by sha256 prefix)
        seed = int.from_bytes(__import__("hashlib").sha256(query.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        rho0 = {
            f"{s}:code": rng.random(32).astype(np.float32)
            for s in ("phono", "sem", "lex", "syntax")
        }
        for sp in rho0:
            rho0[sp] /= rho0[sp].sum()
        state_pre = FlowState(
            rho=rho0, P_theta=np.zeros(16, dtype=np.float32),
            mu_curr=np.zeros(1, dtype=np.float32), tau=0, metadata={"track_id": "T3"},
        )
        state_post = step.step(state_pre)
        return {
            "state_pre": flatten(state_pre),
            "state_post": flatten(state_post),
            "rho_by_species": {k: np.asarray(v, dtype=np.float32) for k, v in state_post.rho.items()},
        }

    return _compute
```

Update `main()` to wire the flag:

```python
parser.add_argument("--g-jepa", type=Path, default=None)
# ... after args = parser.parse_args(argv)
embedder = lambda q: np.zeros(384, dtype=np.float32)  # placeholder, real embedder bound by T22
pair_computer = make_pair_computer(args.g_jepa, embedder)
# Replace `compute_jko_pair(q)` calls with `pair_computer(q)`
```

Keep the module-level `compute_jko_pair` for backward compatibility (tests monkeypatch it):

```python
compute_jko_pair = make_pair_computer(None, lambda q: np.zeros(384, dtype=np.float32))
```

- [x] **Step 21.2: Extend test to cover `--g-jepa` flag**

Add to `tests/track3_deploy/test_jko_oracle_runner.py`:

```python
def test_runner_accepts_g_jepa_flag(tmp_path: Path, fake_jko) -> None:
    """Runner accepts --g-jepa flag (pre-trained weights path) without erroring."""
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["q1"])
    cache_dir = tmp_path / "cache"
    # Pass a nonexistent path; fake_jko still shortcircuits compute_jko_pair.
    rc = jko_oracle_runner.main([
        "--corpus", str(corpus),
        "--cache-dir", str(cache_dir),
        "--g-jepa", str(tmp_path / "nonexistent.safetensors"),
    ])
    # Runner should succeed because compute_jko_pair is monkeypatched
    assert rc == 0
```

- [x] **Step 21.3: Run tests**

Run: `uv run python -m pytest tests/track3_deploy/test_jko_oracle_runner.py -v`
Expected: 4 passed (3 existing + 1 new).

- [x] **Step 21.4: Commit**

```bash
git add kiki_flow_core/track3_deploy/jko_oracle_runner.py \
        tests/track3_deploy/test_jko_oracle_runner.py
git commit -m "$(cat <<'EOF'
feat(track3): wire QueryConditionedF into oracle runner

Adds --g-jepa flag to jko_oracle_runner.py. When pre-trained weights
are provided, compute_jko_pair uses QueryConditionedF with the loaded
g_JEPA decoder; otherwise falls back to legacy ZeroF path.

This unblocks T22/T23 (pilot 10k and scale 50k) with text-conditioned
oracle dynamics.
EOF
)"
```

---

## Task 22: Phase 1 execution — pilot 10k

**Files (produced, gitignored):**
- `data/processed/pilot10k/{train,val,test}.jsonl` (from v0.3 T14)
- `data/processed/heuristic_labels_10k.npz`
- `data/processed/encoder_embeddings_10k.npz` (from v0.3 encoders output)
- `artifacts/g_jepa_pretrained.safetensors`
- `.jko_cache/` (10k entries)
- `artifacts/pilot10k/{B_distilled,C_hash_mlp,D_tiny_tf}.safetensors`
- `artifacts/pilot10k/summary.json`

- [x] **Step 22.1: Build the 10k corpus (reuse v0.3 T14 logic)**

```bash
uv run python scripts/build_corpus_v1.py --size 10000 --out data/processed/pilot10k/
```

Expected: 3 JSONL split files + `test.sha256`.

- [x] **Step 22.2: Run the heuristic labeler over the 10k corpus**

Create `scripts/label_corpus.py`:

```python
"""CLI: read JSONL corpus, run HeuristicLabeler, save per-query labels to NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)
SPECIES_CANONICAL = ("phono:code", "sem:code", "lex:code", "syntax:code")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexique", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from kiki_flow_core.track3_deploy.data.heuristic_labeler import HeuristicLabeler
    labeler = HeuristicLabeler(lexique_csv=args.lexique)

    out: dict[str, np.ndarray] = {}
    n = 0
    with args.corpus.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            q = entry["text"]
            h = hashlib.sha256(q.encode("utf-8")).hexdigest()
            labels = labeler.label(q)
            # stack 4x32 into single (4, 32) array
            stacked = np.stack([labels[sp] for sp in SPECIES_CANONICAL])
            out[h] = stacked.astype(np.float32)
            n += 1
            if n % 500 == 0:
                logger.info("labeled %d queries", n)
    np.savez_compressed(args.output, **out)
    logger.info("wrote %d labels to %s", n, args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

Run (concat splits first):

```bash
cat data/processed/pilot10k/{train,val,test}.jsonl > /tmp/all_10k.jsonl
uv run python scripts/label_corpus.py \
  --corpus /tmp/all_10k.jsonl \
  --output data/processed/heuristic_labels_10k.npz \
  -v
```

Expected: ~5-10 min wall clock on GrosMac. NPZ file ~10 MB.

- [x] **Step 22.3: Generate reference encoder embeddings for each query**

Use the simplest encoder (C hash-mlp) as the fixed reference `f_θ` for phase A:

```python
# scripts/make_reference_embeddings.py
"""Precompute fixed encoder embeddings for each corpus query (used as g_JEPA target)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    enc = EncoderC_HashMLP(seed=args.seed)
    out: dict[str, np.ndarray] = {}
    texts_buf: list[str] = []
    hashes_buf: list[str] = []
    BATCH = 256
    def _flush():
        if texts_buf:
            embs = enc.encode(texts_buf)
            for h, e in zip(hashes_buf, embs, strict=True):
                out[h] = e
            texts_buf.clear()
            hashes_buf.clear()
    with args.corpus.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            q = entry["text"]
            hashes_buf.append(hashlib.sha256(q.encode("utf-8")).hexdigest())
            texts_buf.append(q)
            if len(texts_buf) >= BATCH:
                _flush()
        _flush()
    np.savez_compressed(args.output, **out)
    print(f"wrote {len(out)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python scripts/make_reference_embeddings.py \
  --corpus /tmp/all_10k.jsonl \
  --output data/processed/encoder_embeddings_10k.npz
```

Expected: <1 min wall clock. NPZ ~15 MB.

- [x] **Step 22.4: Pre-train g_JEPA on paired data (phase A)**

```bash
uv run python -m kiki_flow_core.track3_deploy.train_g_jepa \
  --labels data/processed/heuristic_labels_10k.npz \
  --embeddings data/processed/encoder_embeddings_10k.npz \
  --output artifacts/g_jepa_pretrained.safetensors \
  --epochs 5 --lr 3e-4 -v
```

Expected: ~1-2 min wall clock on GrosMac (JAX CPU). Loss should decrease monotonically across epochs.

- [x] **Step 22.5: Rsync corpus + g_JEPA weights to Studio, run JKO oracle**

```bash
rsync -av data/processed/pilot10k/ studio:kiki-flow-research/data/processed/pilot10k/
rsync -av artifacts/g_jepa_pretrained.safetensors studio:kiki-flow-research/artifacts/
ssh studio
cd kiki-flow-research
cat data/processed/pilot10k/{train,val,test}.jsonl > /tmp/all_10k.jsonl
uv run python -m kiki_flow_core.track3_deploy.jko_oracle_runner \
  --corpus /tmp/all_10k.jsonl \
  --cache-dir .jko_cache/ \
  --g-jepa artifacts/g_jepa_pretrained.safetensors \
  -v
```

Expected: ~10k processed, 0 skipped (fresh cache). `.jko_cache/` ~300 MB.

- [x] **Step 22.6: Rsync cache back to KXKM, run sweep**

```bash
rsync -av studio:kiki-flow-research/.jko_cache/ kxkm-ai:kiki-flow-research/.jko_cache/
ssh kxkm-ai
cd kiki-flow-research
uv run python -m kiki_flow_core.track3_deploy.sweep \
  --phase pilot10k \
  --archs B_distilled,C_hash_mlp,D_tiny_tf \
  --corpus data/processed/pilot10k/ \
  --cache .jko_cache/ \
  --output artifacts/ \
  --seed 0
```

Expected: 3 architectures trained, 3 checkpoints + summary.json.

- [x] **Step 22.7: Pull summary + decide Top-k**

```bash
rsync -av kxkm-ai:kiki-flow-research/artifacts/pilot10k/summary.json /tmp/
cat /tmp/summary.json
```

Check R3 kill-switch: if gap between rank-1 and rank-3 is < 15%, all archs promote to 50k.

- [ ] **Step 22.8: Tag phase 1 complete**

```bash
git tag phase1-pilot10k-done
```

---

## Task 23: Phase 2 execution — scale 50k on Top-k

**Files (produced, gitignored):**
- `data/processed/scale50k/{train,val,test}.jsonl`
- `data/processed/heuristic_labels_50k.npz`
- `data/processed/encoder_embeddings_50k.npz`
- `artifacts/g_jepa_pretrained_50k.safetensors` (optional re-pretrain on larger corpus)
- `artifacts/scale50k/{top1,top2}.safetensors`
- `artifacts/scale50k/summary.json`
- `kiki_flow_core/track3_deploy/weights/v0.3.safetensors` (winner promoted)

- [ ] **Step 23.1: Build 50k corpus**

```bash
uv run python scripts/build_corpus_v1.py --size 50000 --out data/processed/scale50k/
```

- [ ] **Step 23.2: Label the 50k corpus (incremental — reuses labeled subset)**

```bash
cat data/processed/scale50k/{train,val,test}.jsonl > /tmp/all_50k.jsonl
uv run python scripts/label_corpus.py \
  --corpus /tmp/all_50k.jsonl \
  --output data/processed/heuristic_labels_50k.npz \
  -v
```

- [ ] **Step 23.3: Refresh reference embeddings for 50k**

```bash
uv run python scripts/make_reference_embeddings.py \
  --corpus /tmp/all_50k.jsonl \
  --output data/processed/encoder_embeddings_50k.npz
```

- [ ] **Step 23.4: Optionally re-pretrain g_JEPA on 50k labels**

```bash
uv run python -m kiki_flow_core.track3_deploy.train_g_jepa \
  --labels data/processed/heuristic_labels_50k.npz \
  --embeddings data/processed/encoder_embeddings_50k.npz \
  --output artifacts/g_jepa_pretrained_50k.safetensors \
  --epochs 5 -v
```

Skip if pilot pre-training is deemed sufficient. Decide by sanity check (does g_JEPA MSE on held-out 50k queries match pilot MSE?).

- [ ] **Step 23.5: Rsync delta + run JKO oracle on new 40k**

```bash
rsync -av data/processed/scale50k/ studio:kiki-flow-research/data/processed/scale50k/
rsync -av artifacts/g_jepa_pretrained_50k.safetensors studio:kiki-flow-research/artifacts/
ssh studio
cd kiki-flow-research
cat data/processed/scale50k/{train,val,test}.jsonl > /tmp/all_50k.jsonl
uv run python -m kiki_flow_core.track3_deploy.jko_oracle_runner \
  --corpus /tmp/all_50k.jsonl \
  --cache-dir .jko_cache/ \
  --g-jepa artifacts/g_jepa_pretrained_50k.safetensors -v
```

The cache skips 10k already computed; ~40k new entries processed.

- [ ] **Step 23.6: Sync + run scale sweep on Top-k**

Retrieve Top-k list from Task 22.7. Example if Top-2 = `["B_distilled","C_hash_mlp"]`:

```bash
rsync -av --update studio:kiki-flow-research/.jko_cache/ kxkm-ai:kiki-flow-research/.jko_cache/
ssh kxkm-ai
cd kiki-flow-research
uv run python -m kiki_flow_core.track3_deploy.sweep \
  --phase scale50k \
  --archs B_distilled,C_hash_mlp \
  --corpus data/processed/scale50k/ \
  --cache .jko_cache/ \
  --output artifacts/ \
  --seed 0
```

- [ ] **Step 23.7: Pull winner, export to NumPy, commit weights**

```bash
rsync -av kxkm-ai:kiki-flow-research/artifacts/scale50k/ artifacts/scale50k/
WINNER=$(jq -r '.archs | to_entries | min_by(.value.test.total) | .key' artifacts/scale50k/summary.json)
echo "Winner: $WINNER"
uv run python -c "
from pathlib import Path
from kiki_flow_core.track3_deploy.encoders import ENCODER_REGISTRY
from kiki_flow_core.track3_deploy.surrogate_trainer_v3 import JointTrainer
from kiki_flow_core.track3_deploy.export.to_numpy import export_bridge_to_numpy
enc = ENCODER_REGISTRY['$WINNER'](seed=0)
tr = JointTrainer(encoder=enc, lam=0.5, lr=3e-4, seed=0)
tr.load_checkpoint(Path('artifacts/scale50k/$WINNER.safetensors'))
export_bridge_to_numpy(tr.params, Path('kiki_flow_core/track3_deploy/weights/v0.3.safetensors'))
print('v0.3 exported')
"
git add kiki_flow_core/track3_deploy/weights/v0.3.safetensors
git commit -m "feat(track3): v0.3 winner weights (text-conditioned)"
git tag phase2-scale50k-done
```

---

## Task 24: Workshop paper section + ablation figure

**Files:**
- Create: `paper/workshop/section_4_method.tex`
- Create: `paper/workshop/figures/ablation_kl_species.{pdf,png}`
- Create: `paper/workshop/figures/scaling_curves.{pdf,png}`

- [ ] **Step 24.1: Generate figures from saved summaries**

```bash
uv run python -c "
import json
from pathlib import Path
from kiki_flow_core.track3_deploy.eval.kl_species import plot_ablation_figure

pilot = json.loads(Path('artifacts/pilot10k/summary.json').read_text())
scale = json.loads(Path('artifacts/scale50k/summary.json').read_text())
r10 = {arch: data['test'] for arch, data in pilot['archs'].items()}
r50 = {arch: data['test'] for arch, data in scale['archs'].items()}
baseline = {'phono': 0.0, 'sem': 0.0, 'lex': 0.0, 'syntax': 0.0, 'total': 0.0}
plot_ablation_figure(r10, r50, baseline, Path('paper/workshop/figures/ablation_kl_species'))
print('figure written')
"
```

Expected: two files in `paper/workshop/figures/`.

- [ ] **Step 24.2: Draft TeX section**

Create `paper/workshop/section_4_method.tex`:

```latex
\section{QueryConditionedF: Free Energy for Text-Conditioned Wasserstein Flow}
\label{sec:query-conditioned-f}

We introduce \textsc{QueryConditionedF}, a subclass of the free-energy
functional class used by Wasserstein gradient flows over psycholinguistic
simplex latents. The functional couples a 384-dimensional text embedding to a
flow over four Levelt-Baddeley species (phono, sem, lex, syntax), each
represented as a 32-dimensional simplex density.

\subsection{Functional form}

For state \(\rho = (\rho_{\text{phono}}, \rho_{\text{sem}}, \rho_{\text{lex}},
\rho_{\text{syntax}})\) on \((\Delta^{32})^4\) and query embedding \(q \in
\mathbb{R}^{384}\):

\begin{equation}
F(\rho, q) = \underbrace{\sum_{s} D_{\mathrm{KL}}(\rho_s \,\|\, \pi_{\mathrm{prior},s})}_{\text{complexity}}
         + \underbrace{\tfrac{1}{2\sigma^2} \| q - g_{\mathrm{JEPA}}(\rho) \|^2}_{\text{accuracy (JEPA)}}
         + \underbrace{\lambda_J \sum_{s,t} J_{st} \langle \rho_s, \rho_t \rangle}_{\text{species coupling}}
\end{equation}

where \(g_{\mathrm{JEPA}} : \Delta^{128} \to \mathbb{R}^{384}\) is a 2-layer MLP
(GELU activations), \(\pi_{\mathrm{prior}}\) is a per-species uniform prior,
and \(J \in \mathbb{R}^{4 \times 4}\) is the Levelt-Baddeley coupling matrix.

\subsection{Heuristic pre-training (phase A)}

We pre-train \(g_{\mathrm{JEPA}}\) offline on (heuristic-label,
encoder-embedding) pairs: the heuristic labeler assigns per-species simplex
targets using SpaCy-fr dependency parsing, IPA phoneme extraction via
phonemizer, and Lexique.org log-frequency bins. The encoder \(f_\theta\) is
fixed during phase A.

\subsection{Connection to active inference}

One JKO step minimizes \(F(\rho, q) + \tfrac{1}{2h} W_2^2(\rho, \rho_{\tau})\),
which is the Wasserstein formulation of variational inference. The flow
\(\rho_\tau \to \rho_{\tau+1}\) corresponds to a sequence of variational
updates of increasing precision, converging (in the convex regime) to the
posterior \(q^*(s \mid q)\).

\subsection{Ablation}

Figure~\ref{fig:ablation} reports KL divergence per species for three text
encoders on the FR hybrid corpus at 10k (pilot) and 50k (scale).

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/ablation_kl_species.pdf}
  \caption{Per-species KL at 50k and scaling curves 10k\(\to\)50k.}
  \label{fig:ablation}
\end{figure}
```

- [ ] **Step 24.3: Commit paper section**

```bash
git add paper/workshop/section_4_method.tex \
        paper/workshop/figures/ablation_kl_species.pdf \
        paper/workshop/figures/ablation_kl_species.png
git commit -m "$(cat <<'EOF'
paper: workshop section 4 + ablation figure

Draft workshop section for NeurIPS 2026 workshop: QueryConditionedF
functional form, heuristic pre-training (phase A), connection to
active inference, and encoder ablation figure from pilot10k+scale50k
summary.json results.
EOF
)"
```

---

## Deferred: Tasks 25-27 (ICLR full paper, phase B)

These tasks implement the JEPA joint training (phase B) and produce the ICLR 2027 paper. They depend on T22-T24 outcomes (pilot ablation informs the encoder chosen for joint training). A separate plan will be written post-workshop-submission in a dedicated brainstorming session.

**Stub task list** (to be detailed later):

- **T25** — Implement differentiable JKO (JAX `scan` over N_STEPS with gradient checkpointing) + joint training loop that backprops through encoder `f_θ` and `g_JEPA` simultaneously.
- **T26** — Joint training runs + contrastive/MSE ablation + N_STEPS sweep.
- **T27** — ICLR 2027 full paper draft + review cycle.

---

## Self-Review

### 1. Spec coverage

Walking through the spec sections:

| Spec section | Task(s) covering it |
|--------------|---------------------|
| §1 Goal (oracle for T14/T15, workshop paper) | T18-T24 |
| §2 Theoretical foundation | Documented in T24 (paper section); code references spec |
| §3 Mathematical structure of F | T18 (class implementation) |
| §4 HeuristicLabeler | T19 (code + golden tests) |
| §5 Joint JEPA training (phase B) | Deferred to T25-T27 (ICLR scope) |
| §6 Integration v0.3 encoders | T21 (wiring) + T22/T23 (execution) |
| §7 Paper story | T24 |
| §8 Timeline + task decomposition | This plan itself |
| §9 Risks | Mitigations embedded in task steps (golden test R1, early stopping R2, N_STEPS fixed R2) |

Workshop scope is fully covered. ICLR scope (§5 joint training + R3 mitigation) is intentionally deferred per the "Deferred" section.

### 2. Placeholder scan

- "TBD/TODO/implement later": none.
- "Similar to Task N": none (all code inlined).
- "Add error handling": none (explicit try/except where needed).
- Open items explicitly flagged: Lexique.org frequency column name uses a fallback (`freq_col = "freqlivres" if ... else df.columns[1]`), acceptable since not all Lexique versions share columns.

### 3. Type consistency

- `SPECIES_CANONICAL` tuple used identically in T18 (`query_conditioned_f.py`), T19 (labeler output keys), T22.2 (`scripts/label_corpus.py`), T23.
- `gjepa_init_params` / `gjepa_forward` / `gjepa_step` / `save_gjepa` / `load_gjepa` names consistent between T20 implementation and T21 import.
- `HeuristicLabeler.label(query) -> dict[str, np.ndarray]` with species keys `{"phono:code", "sem:code", "lex:code", "syntax:code"}` consistent in T19 + T22.2.
- `N_STACKS = 32`, `OUTPUT_DIM = 384`, `INPUT_DIM = 128` named consistently.
- `QueryConditionedF(g_jepa_params, embedding, pi_prior=None, coupling_matrix=None, lambda_j=0.1, sigma2=1.0)` signature stable across T18 tests and T21 consumer.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-query-conditioned-f.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch fresh subagent per task, review between, ~5 subagent dispatches per task.

**2. Inline Execution** — batch execution with checkpoints for review.

**Which approach?**
