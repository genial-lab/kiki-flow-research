---
title: QueryConditionedF — Active-Inference FreeEnergy for Text-Conditioned Wasserstein Flow
date: 2026-04-19
status: approved (brainstorming phase complete, pending user review)
author: L'Electron Rare (clement@saillant.cc)
project: kiki-flow-research
target_release: v0.3 (workshop) + v0.4 (full paper)
related:
  - v0.3 text-bridge surrogate spec (2026-04-19-text-bridge-surrogate-design.md)
  - paper v0.2 (tagged 2026-04-19)
submission_targets:
  - NeurIPS 2026 workshop (sept 2026)
  - ICLR 2027 full paper (sept 2026)
---

# QueryConditionedF — Design Spec

## 1. Goal & Non-Goals

### Goal

Introduire `QueryConditionedF`, une nouvelle sous-classe de `FreeEnergy` dans `kiki_flow_core/` qui couple un embedding texte 384-dim à un flux Wasserstein sur les 4 species Levelt-Baddeley (phono, sem, lex, syntax — 32 stacks chacune). La fonctionnelle est motivée par le principe de minimum free energy variationnelle (Friston 2010+) et opérationnalisée via une architecture JEPA-style (LeCun 2022, Assran et al. 2023) où un prédicteur `g_JEPA: Δ^{128} → ℝ^{384}` apprend à reconstruire l'embedding query depuis l'état species.

**Livrables** :
1. Un oracle JKO query-conditionné pour T14/T15 du sprint v0.3 text-bridge surrogate — remplace `ZeroF`.
2. Un workshop paper (NeurIPS 2026 workshop, deadline sept) centré sur `QueryConditionedF` + heuristic labeler + ablation encoder.
3. Un full paper (ICLR 2027, deadline sept) centré sur le joint JEPA-AIF training via JKO différentiable.

### Non-Goals

- **Pas de remplacement** de `T1FreeEnergy` / `T2FreeEnergy` pour `track1_perf` / `track2_paper` — ils gardent leur rôle.
- **Pas de modèle cérébral unifié** — scope limité à « flux query-conditionné sur simplex psycholinguistique ».
- **Pas de parser FR from scratch** — on réutilise SpaCy-fr, Lexique.org, phonemizer, éventuellement WordNet-fr.
- **Pas d'alignement empirique sur données humaines** (reading-time, fMRI) — restriction à propriétés computationnelles du flow.
- **Pas d'extension multilingue** — FR only pour ces 2 papers.

## 2. Théorie

### Backbone — Variational Free Energy Principle (Friston 2010)

$$
F(q, o) = \underbrace{D_{\mathrm{KL}}(q(s) \,\|\, p(s))}_{\text{complexity}} - \underbrace{\mathbb{E}_{q(s)}[\log p(o \mid s)]}_{\text{accuracy}}
$$

où :
- `s` = species state (simplex sur 4 × 32)
- `o` = query embedding (sortie de l'encodeur text)
- `q(s)` = notre `ρ` courant (posterior approximé par le flow)
- `p(s)` = prior sur species states (π_prior, Levelt-Baddeley YAML ou uniforme)
- `p(o|s)` = likelihood **implémenté via JEPA** : `g_JEPA(ρ) → embedding prédit`, gaussian likelihood

### Rôle de JEPA

JEPA (Joint Embedding Predictive Architecture, LeCun 2022) fournit le scaffold self-supervised pour entraîner `g_JEPA`. Dans le langage JEPA standard :
- **Encodeur** `f_θ` : text → latent embedding (= notre encodeur v0.3, T5/T6/T7)
- **Predictor** `g_JEPA` : ρ → predicted embedding (= notre decoder)
- **Target** : self-référence JEPA, `f_θ(query)` est ce que `g_JEPA(ρ_posterior(query))` doit prédire
- **Loss** : `||f_θ(query) - g_JEPA(ρ_posterior)||²` en latent space (pas de reconstruction pixel-level)

JEPA réalise l'opération de restructure (Pillar D/FEP) sans primitive de replay explicite. Référence : LeCun 2022 (*A Path Towards Autonomous Machine Intelligence*), Assran et al. 2023 (*I-JEPA*), Bardes et al. 2024 (*V-JEPA*).

### Levelt-Baddeley species comme latents

Les 4 species (phono, sem, lex, syntax) sont des **variables latentes cognitivement interprétables** du générative model AIF. Chacune compte 32 stacks (activation channels) :
- **phono** : classes phonétiques (voyelles ouvertes/fermées/nasales, stops vl/vd, fricatives, ...)
- **sem** : domaines sémantiques top-level (person, place, artifact, action, attribute, ...)
- **lex** : bins de log-fréquence Lexique.org
- **syntax** : patterns syntaxiques FR fréquents

### Connexion JKO ↔ inférence variationnelle

Une étape JKO avec `F` est une étape de mise à jour variationnelle. JKO minimise le lagrangien :

$$
\rho_{τ+1} = \arg\min_{\rho} \left[ F(\rho, q) + \frac{1}{2h} W_2^2(\rho, \rho_τ) \right]
$$

qui est la formulation Wasserstein de la variational update en active inference. Le flow τ → τ+1 correspond à une séquence d'inférences variationnelles de plus en plus précises, convergeant (quand convexe) vers le posterior AIF `q*(s)` qui minimise `F`.

## 3. Structure mathématique de F(ρ, q)

$$
F(\rho, q) = \underbrace{\sum_{s} D_{\mathrm{KL}}(\rho_s \,\|\, \pi_{\mathrm{prior},s})}_{\text{(a) complexity}} \;+\; \underbrace{\frac{1}{2\sigma^2} \| \mathrm{embed}(q) - g_{\mathrm{JEPA}}(\rho) \|^2}_{\text{(b) accuracy (JEPA likelihood)}} \;+\; \underbrace{\lambda_J \sum_{s,t} J_{st}\, \langle \rho_s, \rho_t \rangle}_{\text{(c) species coupling (T2-style)}}
$$

### Termes

**(a) Complexity per-species.** `Σ_s KL(ρ_s || π_prior_s)`. Prior par species depuis `kiki_flow_core/species/data/levelt_baddeley_coupling.yaml` (ou uniform 1/32 en v1). Gradient analytique :

$$\frac{\partial F_{(a)}}{\partial \rho_s}(k) = \log\frac{\rho_s(k)}{\pi_{\mathrm{prior},s}(k)} + 1$$

**(b) Accuracy via JEPA likelihood.** `||embed(q) − g_JEPA(ρ)||² / (2σ²)` avec `σ=1.0` par défaut. `g_JEPA: ℝ^{128} → ℝ^{384}` = MLP 2 couches (128→256→384, GELU). Gradient via autodiff JAX :

$$\frac{\partial F_{(b)}}{\partial \rho_s} = -\frac{1}{\sigma^2} \left( \mathrm{embed}(q) - g_{\mathrm{JEPA}}(\rho) \right)^\top \frac{\partial g_{\mathrm{JEPA}}}{\partial \rho_s}$$

**(c) Species coupling.** `λ_J Σ_{s,t} J_{st} ⟨ρ_s, ρ_t⟩`. Réutilise `J` du T2FreeEnergy (`MixedCanonicalSpecies.coupling_matrix()`). Gradient analytique :

$$\frac{\partial F_{(c)}}{\partial \rho_s} = 2 \lambda_J \sum_{t} J_{st} \rho_t$$

### Hyperparamètres par défaut

| Paramètre | Valeur v1 | Valeur v2 | Ablation |
|-----------|-----------|-----------|----------|
| σ² (JEPA likelihood noise scale) | 1.0 | 1.0 | yes (0.1, 1.0, 10.0) |
| λ_J (species coupling strength) | 0.1 | 0.1 | yes (0, 0.05, 0.1, 0.5) |
| π_prior_s | uniforme 1/32 | YAML Levelt-Baddeley | yes |
| g_JEPA arch | 128→256→384 GELU | same | — |
| N_STEPS (JKO step depth) | 10 | 10 | yes (5, 10, 20, 50) |

### Invariants (respecte CLAUDE.md)

- Clip `ρ` à `[1e-12, ∞)` avant tout `log`.
- Renormaliser après chaque gradient step (respect simplex, sum=1 ± 1e-4).
- Sinkhorn log-domain si `W²` prox utilisé (`method="sinkhorn_log"`).

### Propriétés limites (pour théorèmes/sanity checks du paper)

| Cas limite | F réduit à | Comportement attendu |
|------------|-----------|---------------------|
| `g_JEPA ≡ 0` et `λ_J = 0` | `Σ_s KL(ρ_s \|\| π_prior)` | flow converge vers `π_prior` (pas de query effect) |
| `π_prior = uniform` et `λ_J = 0` | `H_entropie + ||g_JEPA(ρ)||²` | flow = inverse de `g_JEPA` (JEPA-only) |
| `q = 0` (pas de query) | `H_entropie + ||g_JEPA(ρ)||²` | flow pousse ρ vers kernel de `g_JEPA` |
| `λ_J → ∞` | coupling-dominated | flow suit T2FreeEnergy dynamics |

**Convexité.** `F_(a)` et `F_(c)` sont convexes en ρ. `F_(b)` est convexe si `g_JEPA` est linéaire, non-convexe avec MLP. **F est non-convexe globalement** — flow peut avoir local minima. Limitation documentée dans paper.

## 4. HeuristicLabeler pipeline (phase A de pre-training)

### Entrée / sortie

- **Entrée** : query FR texte (str).
- **Sortie** : `π_target ∈ Δ^{4 × 32}` — 4 species, 32-dim simplex chacune (sum=1).

### Per-species feature extraction

| Species | 32 stacks représentent | Outils FR | Projection |
|---------|------------------------|-----------|------------|
| **phono:code** | Classes phonétiques (4×8 = 32) | `phonemizer` (espeak-ng FR backend) | Count phonemes per class → normalize |
| **sem:code** | Domaines sémantiques top-level (WordNet-fr ou custom 32 categories) | SpaCy-fr NER + lemmatizer | tf-idf-ish over domains for content words |
| **lex:code** | 32 bins log-frequency Lexique.org | Lexique.org 3.83 CSV | Histogramme normalisé des tokens par bin |
| **syntax:code** | 32 patterns syntaxiques FR fréquents | SpaCy-fr dependency parse | Proportion of tokens involved per pattern |

### Module

Chemin : `kiki_flow_core/track3_deploy/data/heuristic_labeler.py`

```python
class HeuristicLabeler:
    def __init__(
        self,
        lexique_csv: Path,
        spacy_model: str = "fr_core_news_lg",
        sem_categories: list[str] | None = None,
        syntax_patterns: list[str] | None = None,
    ) -> None:
        self._nlp = spacy.load(spacy_model)
        self._phonemizer = Phonemizer(language="fr-fr", backend="espeak-ng")
        self._lexique = pd.read_csv(lexique_csv)
        self._sem_categories = sem_categories or _DEFAULT_SEM_CATEGORIES
        self._syntax_patterns = syntax_patterns or _DEFAULT_SYNTAX_PATTERNS

    def label(self, query: str) -> dict[str, np.ndarray]:
        """Return {'phono:code': (32,), 'sem:code': (32,), 'lex:code': (32,), 'syntax:code': (32,)}, each simplex."""
        doc = self._nlp(query)
        return {
            "phono:code": self._phono_distribution(query),
            "sem:code": self._sem_distribution(doc),
            "lex:code": self._lex_distribution(doc),
            "syntax:code": self._syntax_distribution(doc),
        }
```

### Offline corpus labeling

Script : `scripts/label_corpus.py` consomme un corpus JSONL, produit un `.npz` avec `{query_hash: π_target}` indexé SHA256(query). Permet caching + audit.

### Golden test

20 queries annotées manuellement. Exemples :
- Rhyming nursery rhyme → phono peaked, low syntax complexity
- Abstract nouns query → sem peaked, low phono specificity
- Complex subordinated sentence → syntax peaked
- Technical jargon → lex peaked on rare bins

Teste que `HeuristicLabeler.label()` correspond aux intuitions linguistiques ± 20%.

### Effort estimé

~3-4 jours de vrai NLP FR (SpaCy-fr loading, Lexique CSV mapping, phonemizer espeak-ng config, 32-class phonetic mapping, WordNet-fr access ou custom OntoFR fallback).

## 5. Joint JEPA training (phase B)

### Architecture JEPA-style

- **Encoder `f_θ`** (text → 384-dim) : l'encodeur v0.3 (gagnant de l'ablation, ou T5 hash-mlp fixé en v1 pour simplicité).
- **Predictor `g_JEPA`** (ρ → 384-dim) : MLP 128→256→384 (GELU), les paramètres de notre F.
- **Target** : self-référence JEPA, `f_θ(query)` est le target que `g_JEPA(ρ_posterior(query))` doit prédire.

### Phase A — Pre-train `g_JEPA` seul (~5k queries, ~1h GPU)

```python
# Pseudo-code
for query in corpus:
    π_target = heuristic_labeler.label(query)
    target_emb = f_θ_frozen(query)  # fixed reference encoder (T5 random init par ex.)
    ρ_flat = jnp.concatenate([π_target[sp] for sp in SPECIES_CANONICAL])  # (128,)
    loss = ((target_emb - g_JEPA(ρ_flat))**2).mean()
    g_JEPA.params -= lr * jax.grad(loss_fn)(g_JEPA.params)
# Freeze g_JEPA after convergence, save weights
```

Sortie : `artifacts/g_JEPA_pretrained.safetensors`.

### Phase B — Joint self-supervised training (~20 epochs, 10k corpus, ICLR scope)

```python
# Pseudo-code; JAX scan over N_STEPS JKO steps, differentiable end-to-end
def full_loss(f_θ_params, g_JEPA_params, query):
    q_emb = f_θ(query, f_θ_params)  # (384,)
    # Initialize ρ uniform, run N_STEPS JKO steps with QueryConditionedF(g_JEPA, q_emb)
    ρ_init = jnp.ones((4, 32)) / 32
    def jko_step(ρ, _):
        grad_F = compute_grad_F(ρ, q_emb, g_JEPA_params, π_prior, J, λ_J)
        ρ_new = ρ - step_size * grad_F
        ρ_new = jnp.maximum(ρ_new, 1e-12)
        ρ_new = ρ_new / ρ_new.sum(axis=-1, keepdims=True)
        return ρ_new, None
    ρ_posterior, _ = jax.lax.scan(jko_step, ρ_init, None, length=N_STEPS)
    # Loss: JEPA prediction error at convergence
    return ((q_emb - g_JEPA(flatten(ρ_posterior), g_JEPA_params))**2).mean()

# Update both encoder and g_JEPA via autodiff
grads = jax.grad(full_loss, argnums=(0, 1))(f_θ.params, g_JEPA.params, query_batch)
```

### Contrastive variant (JEPA-pure, ablation dim)

Au lieu de MSE single-pair, InfoNCE sur un batch : `(q_i, ρ_posterior(q_i))` pull together, `(q_i, ρ_posterior(q_j))` push apart pour `i ≠ j`. Plus JEPA-authentique mais ajoute complexité (batch de négatifs, température). Ablation pour ICLR paper.

### Défis techniques

- **Backprop à travers `jax.lax.scan` sur 10 steps JKO** : graph deep mais JAX-differentiable. Mémoire O(N_STEPS × batch × 128). Gradient checkpointing si OOM.
- **Convergence du flow pendant training** : risque d'instabilité si ρ oscille. Fix N_STEPS low (5-10 en v1), monitor stability.
- **LR schedule** : LR séparé pour `f_θ` (small, pre-trained) vs `g_JEPA` (larger, freshly optimized). AdamW avec cosine schedule.

## 6. Intégration avec v0.3 sprint (encoders B/C/D)

Le sprint v0.3 text-bridge surrogate a 13 tâches déjà complètes (T1-T13). Comment s'articule avec `QueryConditionedF` ?

### Proposition — 2 phases paper, 2 contributions distinctes

**Phase 1 (workshop paper, NeurIPS 2026 sept)** :
- **Contribution** : `QueryConditionedF` + heuristic labeler comme oracle query-conditionné.
- Le sprint v0.3 déjà-fait tourne **avec cet oracle** (au lieu de `ZeroF`) → T22/T23 (remplacent T14/T15) produisent des vrais pairs text-conditionnés.
- Encoder ablation (B/C/D) mesure quelle architecture capture le mieux le signal query→flow généré par `QueryConditionedF`.
- Paper story : « pour tester un bridge surrogate text-conditionné, on a besoin d'un oracle text-conditionné ; voici un design principled via AIF+labeler+Levelt-Baddeley ».
- `g_JEPA` est **fixé** (pre-trained phase A, gelé). **Pas de joint training**.

**Phase 2 (full paper, ICLR 2027 sept)** :
- **Contribution** : joint JEPA training sur le gagnant d'ablation de Phase 1.
- Phase 1 identifie le gagnant (probablement C hash-mlp ou D tiny-tf) → devient `f_θ`.
- Phase B du joint training démarre : `f_θ` + `g_JEPA` learned jointly via flow backprop.
- Paper story : « self-supervised refinement of the text encoder and its reverse decoder via the Wasserstein flow they generate ; new application of JEPA to simplex latents ».
- Ablations : joint vs frozen `g_JEPA`, contrastive vs MSE loss, N_STEPS flow depth.

### Avantages de ce découpage

- Les 13 tâches v0.3 déjà-faites sont **préservées** et gagnent un vrai oracle.
- Contribution workshop (small but solid) ≠ contribution ICLR (deeper). Risque de refus réduit.
- Timeline cohérente : phase 1 livrable en ~4 semaines, phase 2 a ~3-4 mois supplémentaires.

## 7. Paper story final

### Workshop paper — NeurIPS 2026 (deadline sept)

**Titre candidat** : *« QueryConditionedF: Active-Inference FreeEnergy for Text-Conditioned Wasserstein Flow on Psycholinguistic Simplex Latents »*

**Abstract structure** (~200 mots) :
1. **Motivation** : text-conditioned dynamics need grounded oracles. Classical FreeEnergy (ZeroF, T2) are query-agnostic.
2. **Contribution** : `QueryConditionedF` couples text embeddings to Wasserstein gradient flow over 4 Levelt-Baddeley species via AIF (complexity + JEPA likelihood + species coupling).
3. **Method** : pre-train JEPA predictor `g_JEPA` via heuristic FR labeler.
4. **Experiments** : ablation of 3 text encoders (B distilled MiniLM, C hash-mlp, D tiny-transformer) on FR hybrid corpus (50k) ; KL-per-species metrics ; scaling 10k → 50k.
5. **Conclusion** : `QueryConditionedF` enables principled encoder ablation ; heuristic labeler is reproducible.

**Contributions** (ordre) :
1. Design `QueryConditionedF` avec gradients analytiques.
2. `HeuristicLabeler` FR (pipeline reproducible SpaCy + Lexique + phonemizer).
3. Ablation encoder B/C/D démontre que l'architecture impacte la capture du coupling query→flow.
4. Validation : pairs générés sont text-conditionnés (vs ZeroF baseline qui ne l'est pas).

**Format** : ~4-6 pages + appendix, 2 figures ablation + 1 architecture diagram.

### Full paper — ICLR 2027 (deadline sept)

**Titre candidat** : *« Self-Supervised Refinement of Text Encoders via Wasserstein Gradient Flow: JEPA Meets Active Inference on Simplex Latents »*

**Abstract structure** (~250 mots) :
1. **Motivation** : JEPA pre-trains encoders in latent space via self-prediction. We extend JEPA to simplex-constrained latent spaces via AIF flow.
2. **Contribution** : joint training of encoder `f_θ` and JEPA predictor `g_JEPA` via backprop through JKO steps.
3. **Theory** : fixed-point analysis, convergence conditions, connection to variational inference.
4. **Method** : differentiable JKO via `jax.lax.scan`, N_STEPS depth analysis, contrastive vs MSE loss comparison.
5. **Experiments** : multi-seed FR corpus, baselines (frozen `g_JEPA`, random `g_JEPA`, classical JEPA), ablations on N_STEPS and loss form.
6. **Conclusion** : self-supervised refinement on simplex latents is feasible and outperforms frozen baselines.

**Contributions** :
1. Joint JEPA-AIF training via JKO différentiable.
2. Contrastive vs MSE comparaison en simplex-constrained JEPA.
3. Analyse N_STEPS de profondeur du flow (théorie + empirique).
4. Multi-seed robustness sur corpus FR.

**Format** : ~10 pages + appendix, 4-5 figures.

### Cohérence narrative

Le workshop établit la fondation (`QueryConditionedF` comme oracle), le full paper étend vers l'auto-supervision (JEPA joint). Les deux papers partagent la même F mais divergent sur le régime d'entraînement.

## 8. Timeline & task decomposition

### Tâches additionnelles (sur les 4 restantes de v0.3 — T14/T15/T16/T17)

| ID | Task | Effort | Dependencies | Scope |
|----|------|-------:|--------------|-------|
| **T18** | `QueryConditionedF` class (value + grad_rho analytical, autodiff JEPA term) | ~1 j | — | workshop |
| **T19** | `HeuristicLabeler` FR (phono + sem + lex + syntax) + golden tests | ~4 j | — | workshop |
| **T20** | Pre-train `g_JEPA` sur heuristic labels (phase A), save weights | ~1 j + 1h compute | T18, T19 | workshop |
| **T21** | Wire `QueryConditionedF(g_JEPA_pretrained)` dans `jko_oracle_runner` (update T11) | ~0.5 j | T20 | workshop |
| **T22** | Phase 1 pilot 10k avec vrai oracle (replaces T14) | ~1 j + ~1-2h compute | T21, T12, T13 | workshop |
| **T23** | Phase 2 scale 50k sur Top-2 (replaces T15) | ~0.5 j + ~3-4h compute | T22 | workshop |
| **T24** | Workshop paper : section 4 QueryConditionedF + ablation figures | ~1 sem | T23 | workshop |
| **T25** | Joint JEPA training implementation (phase B, JAX scan JKO) | ~2 sem | T24 | **ICLR only** |
| **T26** | Joint training run + ablations (contrastive, N_STEPS sweep) | ~3 sem | T25 | **ICLR only** |
| **T27** | ICLR full paper draft + review cycle | ~3 sem | T26 | **ICLR only** |

### Milestones

- **Workshop NeurIPS 2026** : T18-T24 → **~4-5 semaines** from today (date cible ~mi-mai 2026).
- **Full ICLR 2027** : T25-T27 → **+10-12 semaines** après workshop (date cible ~août 2026, confortable avant deadline sept).

### Gestion des tâches T14/T15 existantes

Deux options :
1. **Update descriptions** de T14/T15 pour pointer vers T22/T23 (garde les IDs).
2. **Mark deleted** et créer T22/T23 neufs.

Recommandation : option 2 (deleted + neufs) pour clarté historique.

### Livrables globaux

1. **Code** : module `kiki_flow_core/track3_deploy/query_conditioned_f.py`, `data/heuristic_labeler.py`, training scripts, updated `jko_oracle_runner.py`.
2. **Weights** : `artifacts/g_JEPA_pretrained.safetensors` (phase A), `artifacts/g_JEPA_joint_trained.safetensors` (phase B, ICLR).
3. **Datasets** : `data/processed/heuristic_labels.npz` (cache labels), `data/processed/corpus_hybrid_v1.jsonl` (reuse v0.3).
4. **Papers** : `paper/workshop-neurips2026.tex` (workshop), `paper/iclr2027.tex` (full).
5. **Figures** : ablation bars, scaling curves, JEPA architecture diagram, N_STEPS convergence curves.
6. **Spec docs** : ce fichier + updates à `2026-04-19-text-bridge-surrogate-design.md`.

## 9. Risks & open questions

| ID | Risque | Impact | Mitigation |
|----|--------|--------|------------|
| **R1** | Labeler heuristique produit labels bruités / arbitraires → g_JEPA apprend garbage | **Critique** | Golden test 20 queries annotées manuellement, audit distribution par species sur corpus pilot 1k avant pre-training |
| **R2** | Non-convexité F → flow a des minima locaux, JKO oscille | Important | Fix N_STEPS=10, damping step size, multi-seed empirical check, early stopping si KL_val remonte |
| **R3** | Backprop à travers JKO steps (phase B) coûteux mémoire/compute | Important | JAX `scan` avec gradient checkpointing, test N_STEPS=5 d'abord, batch petit (32) |
| **R4** | SpaCy-fr 500MB + WordNet-fr installation friction | Minor | Optional deps group `text-bridge-labeler`, doc install explicite, fallback sem clustering KMeans si WordNet-fr absent |
| **R5** | Contrastive JEPA variant complexity (négatifs, température) | Minor | Start MSE, contrastive seulement pour ICLR cycle, pas workshop |
| **R6** | Cohérence narrative entre workshop et ICLR papers | Important | Écrire les 2 abstracts **avant** T18 pour fixer les arcs ; partager figure architecture commune |
| **R7** | Convergence théorique du flow non garantie (non-convexité MLP JEPA) | Important | Citer AIF convergence results (Da Costa 2020, Parr & Friston 2017), + empirical multi-seed, documenter comme limitation |
| **R8** | WordNet-fr licence ambiguë (Creative Commons BY-SA or restricted?) | Minor | Fallback : domains via clustering KMeans sur embeddings MiniLM, cite comme « semi-heuristic » |
| **R9** | Pre-trained `g_JEPA` sous-performe si labeler heuristique est mal aligné avec vrais embeddings | Important | Plot curves (loss vs epoch) sur validation held-out, stop pre-training si plateau ; itérer labeler config si besoin |
| **R10** | Compute budget : joint training ~3 semaines sur KXKM 4090 peut être bloqué par autres projets | Important | Coordination infra en amont, gating mémoire, fallback sur Studio MLX si KXKM unavailable |

### Questions ouvertes à trancher avant T18

1. **Timeline confirmation** : workshop NeurIPS 2026 deadline = ~sept, ICLR 2027 deadline = ~sept (à vérifier les dates exactes).
2. **Budget compute phase B** : KXKM 4090 alloué pour ~3 semaines consecutive ? Ou plusieurs windows ? Impacte scheduling.
3. **Labels heuristiques caching** : stockage .npz sur GrosMac puis rsync vers Studio/KXKM, ou partage via NAS ?
4. **Publication stratégie** : workshop puis full (workshop accepté avant full submission), ou parallèle (risque de double submission) ?
5. **Co-authors** : paper single-author ou collaboration (mentor, collaborateurs externes) ? Affecte scope & rigueur.

---

## Appendix — Brainstorming decisions trace

Pour audit future : les 6 questions de brainstorming et les choix utilisateur (2026-04-19).

| Q | Sujet | Choix retenu |
|---|-------|--------------|
| 1 | Scope / rigueur scientifique | **C** — modèle cognitif complet, contribution paper majeure |
| 2 | Cadre théorique de fond | **E + F + JEPA** — hybride AIF + Levelt-Baddeley + bespoke, avec JEPA (LeCun 2022, Assran 2023, Bardes 2024) comme substrate pour restructure operation (Pillar D/FEP) |
| 3 | Terme mathématique central | **③** — Full AIF + JEPA likelihood + species coupling (V+π+J) |
| 4 | Paramétrisation `g_JEPA` | **B + D** — MLP 2-layer pre-trained offline **puis** jointly-trained JEPA-style |
| 5 | Scope B+D dans v0.3 | **C** — full B+D dans v0.3, décale T14/T15 de 2 semaines, B+D devient cœur du paper |
| 6 | Cible soumission | **B + C** — ICLR 2027 (full, ~5 mois) **et** workshop NeurIPS 2026 (intermediate milestone, sept) |

**Décisions supplémentaires** :
- Split papers : workshop = `QueryConditionedF` + heuristic labeler + encoder ablation (phase 1), full = joint JEPA training (phase 2).
- `g_JEPA` est gelé en phase 1 (workshop), joint-trained en phase 2 (ICLR).
- N_STEPS=10 JKO par défaut, ablation sur {5, 10, 20, 50}.
- Hyperparams v1 : σ²=1.0, λ_J=0.1, π_prior=uniforme ; v2 : YAML Levelt-Baddeley.
- Contrastive JEPA variant reporté à ICLR cycle seulement.
