---
title: Text-Native Bridge Surrogate — Design Spec
date: 2026-04-19
status: approved (brainstorming phase complete, pending user review)
author: L'Electron Rare (clement@saillant.cc)
project: kiki-flow-research
target_release: v0.3
related: paper v0.2 (tagged 2026-04-19), v0.2-d128.safetensors (current bridge surrogate)
---

# Text-Native Bridge Surrogate — Design Spec

## 1. Goal & Non-Goals

### Goal

Éliminer les deux « hand-crafted paths » du bridge surrogate actuel :

- **(a)** le fallback SHA256 dans `QueryEncoder._encode_raw()` (embedding déterministe non-sémantique utilisé quand `sentence-transformers` est indisponible ou `use_stub=True`) ;
- **(b)** l'entraînement du surrogate v0.2-d128 sur **100 paires JKO synthétiques sans contenu texte** — le modèle apprend uniquement la dynamique de flux, pas le couplage sémantique-flux.

Le remplacement est un pipeline **end-to-end texte→delta** où trois architectures d'encodeur (MiniLM distillé, hash+MLP, tiny-transformer) sont comparées via un **sweep ablatif** sur un corpus hybride (psycholinguistique + généraliste FR + synthétique Qwen). Livrables : (i) une figure d'ablation paper-grade avec KL-par-species et scaling 10k→50k sur Top-2, (ii) un encodeur gagnant packagé, (iii) une section paper v0.3 formalisée.

### Non-Goals

- **Pas de refactor du JKO oracle** lui-même. Extension minimale : exposer `rho_by_species` séparément (dict `{species: rho}`) si l'API actuelle aplatit en 128-dim.
- **Pas de remplacement** de `KikiFlowBridge` ni `streaming_runner`. L'interface publique reste identique ; seul l'encodeur et les weights du bridge head sont swappés.
- **Pas d'optimisation déploiement pure-NumPy pendant le sweep**. Runtime JAX/PyTorch autorisé en phase training. Conversion NumPy uniquement en phase packaging du gagnant.
- **Pas d'extension multilingue**. FR only pour cette itération.
- **Pas de changement de scope corpus** au-delà de B+C+D (psycholinguistique + généraliste FR + synthétique Qwen).

## 2. Architecture vue d'ensemble

```
┌──── Corpus Hybride (50k queries FR) ──────────────┐
│  B. Psycho (CHILDES-fr, Lexique, Levelt-Baddeley) │   20% = 10k
│  C. Généraliste FR (OSCAR-fr, Wikipedia FR)       │   40% = 20k
│  D. Synthétique Qwen3.5-35B @ KXKM                │   40% = 20k
└──────────────┬────────────────────────────────────┘
               │
        [JKO Oracle @ Studio MLX]
               │ consomme query → produit
               │ (state_pre, state_post, rho_by_species)
               │
     paires (query_text, state_pre, state_post, rho_by_species)
     └── cachées en .safetensors indexées SHA256(query) ──┘
               │
        ┌──────┴─────────┐
        │                │
   [Train @ KXKM 4090]   │  3 encodeurs en parallèle (time-shared JAX) :
        │                │    (B) MiniLM-distilled   → 384-dim
        │                │    (C) hash+MLP           → 384-dim
        │                │    (D) tiny-transformer   → 384-dim
        │                │  + BridgeHead partagé (re-trained conjointement avec chaque encodeur)
        │                │
   [Eval @ GrosMac] ←────┘  KL-par-species, MAPE_Δ, hit@5, latency, size
        │
   Ranking Top-2 ──────────→ relance scale 50k sur Top-2
        │
   Figure paper + packaging (export NumPy pour le gagnant seulement)
```

**Flux de données principal.**

1. Assemblage corpus (GrosMac) → rsync vers Studio + KXKM.
2. Génération synthétique D via Qwen3.5-35B tunnel (KXKM).
3. Dedup cross-source + split train/val/test.
4. Phase 0 : smoke test JKO oracle (valide `rho_by_species` shape `(4, 32)`).
5. Phase 1 : génération paires JKO 10k (Studio) + training pilote 3 archis (KXKM).
6. Éval pilote (GrosMac) → Top-2.
7. Phase 2 : génération paires JKO +40k (Studio, incrémental avec cache SHA256) + training scale 50k Top-2 (KXKM).
8. Éval scale (GrosMac) → winner final + figures + packaging NumPy.

## 3. Composants

| Composant | Rôle | Localisation | Nouveau / Réutilisé |
|-----------|------|--------------|---------------------|
| `CorpusBuilder` | Assemble B+C+D, dedup exact + embedding, split stratifié | `kiki_flow_core/track3_deploy/data/corpus_builder.py` | **Nouveau** |
| `SyntheticGenerator` | Prompt Qwen3.5-35B via tunnel `localhost:18000`, species-aware | `track3_deploy/data/synth_qwen.py` | **Nouveau** |
| `JKOOracle` | Expose `oracle(query) → (state_pre, state_post, rho_by_species)` | Wrapper léger sur solveur JKO existant | **Extension** |
| `JKOCache` | Cache .safetensors indexé par SHA256(query) pour incrémental | `track3_deploy/data/jko_cache.py` | **Nouveau** |
| `EncoderB_DistilledMiniLM` | MLP ~2M params imitant MiniLM, loss = MSE vs MiniLM target | `track3_deploy/encoders/distilled.py` | **Nouveau** |
| `EncoderC_HashMLP` | Style fastText : n-gram hash → embedding table → MLP, ~520K params | `track3_deploy/encoders/hash_mlp.py` | **Nouveau** |
| `EncoderD_TinyTransformer` | 4-6 layers, 8 heads, ~8M params, JAX natif | `track3_deploy/encoders/tiny_tf.py` | **Nouveau** |
| `BridgeHead` | MLP 512→256→256→128 (tanh), identique architecture à v0.2 | `track3_deploy/neural_surrogate.py` | **Réutilisé** |
| `JointTrainer` | Train (encoder + BridgeHead) end-to-end avec loss MSE + λ·KL | `track3_deploy/surrogate_trainer_v3.py` | **Nouveau (v3)** |
| `SweepRunner` | Orchestre 3 archis, logs W&B/Langfuse, Top-2 gating | `track3_deploy/sweep.py` | **Nouveau** |
| `EvalKL` | KL-par-species + scaling curves + figure matplotlib | `track3_deploy/eval/kl_species.py` | **Nouveau** |
| `NumpyExporter` | Export gagnant en .safetensors + forward pure-NumPy | `track3_deploy/export/to_numpy.py` | **Nouveau** |

## 4. Corpus hybride

### Volumes et sources

| Source | Pilote (10k) | Scale (50k) | Sourcing | Licence |
|--------|-------------:|------------:|----------|---------|
| **B. Psycholinguistique** | 2k | 10k | CHILDES-fr (child/adult speech), Lexique.org (norms lexicales FR), stimuli Levelt-Baddeley (naming tasks, picture descriptions) | CC-BY-NC / académique |
| **C. Généraliste FR** | 4k | 20k | OSCAR-fr échantillon stratifié par longueur (8-64 tokens), Wikipedia FR intro sentences | CC-BY-SA |
| **D. Synthétique Qwen** | 4k | 20k | Qwen3.5-35B via tunnel autossh `localhost:18000`, 4 prompts species-aware × {1k pilot, 5k scale} each, temp=0.8, top_p=0.9 | Interne |

### Species-aware prompts (source D)

4 prompts structurés, un par species Levelt-Baddeley :

```
phono:   "Génère une query courte en français qui sollicite fortement le traitement phonologique (mots rares,
          homophones, assonances, phonèmes difficiles). Longueur 8-24 mots. Une query par ligne."

sémant:  "Génère une query impliquant désambiguïsation sémantique, polysémie, relations lexicales (synonymes,
          hyperonymes, antonymes). Longueur 8-24 mots."

lexical: "Génère une query avec mots de basse fréquence, néologismes récents, registre spécialisé (technique,
          littéraire, scientifique). Longueur 8-24 mots."

syntax:  "Génère une query avec structure syntaxique complexe : dépendances longues, subordonnées imbriquées,
          ambiguïtés d'attachement. Longueur 12-32 mots."
```

Batch de 50 queries par requête Qwen. Parsing : une query par ligne, strip markers numériques.

### Pipeline dedup

1. **Exact match** sur string normalisée (lowercase, strip punct, collapse whitespace).
2. **Embedding dedup** : MiniLM embeddings, cosine similarity > **0.92** → drop le plus court des deux.
3. **Cross-source dedup** : D candidates testées contre (B ∪ C) au seuil 0.92 ; match → drop D.
4. **Audit manuel** : 200 paires (val+test) échantillonnées, validation que < 2 % sont sémantiquement quasi-identiques. Si > 2 %, baisser seuil à 0.88 et re-run.

### Split

Stratifié par (source × species) : **train 80% / val 10% / test 10%**. Test set **gelé avant pilote** (SHA256 du fichier committé en tag `corpus_v1_test`).

## 5. Training protocol

### Phase 0 — Pilote JKO (smoke test, ~5 min)

- 100 queries variées (stratifiées species) → oracle JKO.
- Validation : `rho_by_species` sérialisé en tenseur `(4, 32)` (pas juste un vecteur aplati 128-dim).
- **Kill-switch** : si l'oracle ne produit pas les 4 species séparées, patch nécessaire sur `oracle()` avant de continuer. Aucune autre phase ne démarre tant que Phase 0 ne passe pas.

### Phase 1 — Pilote 10k (KXKM 4090, ~1-2 h)

Pour chaque encodeur ∈ {B, C, D} :

- Train jointly `encoder + BridgeHead` sur 8k train.
- Loss : `L = MSE(Δ_pred, Δ_target) + λ · KL_total(ρ_pred, ρ_target)` avec **λ = 0.5** (non swepé dans v1 ; documenté en R7 comme future work possible).
- Optim : AdamW, warmup 500 steps, cosine decay.
  - B : lr = 3e-4, batch = 128, epochs max = 20.
  - C : lr = 3e-4, batch = 128, epochs max = 20.
  - D : lr = 1e-4, batch = 64, epochs max = 30.
- Early stop sur `KL_total(val)`, patience 3 epochs.
- Checkpoint best val-KL par archi → `artifacts/pilot_10k/{B,C,D}.safetensors`.
- Éval sur test set → KL_total test + KL-par-species + metrics secondaires → **ranking**.

### Phase 2 — Scale 50k sur Top-2 (KXKM, ~3-4 h)

- Top-2 archis retenues → re-train **from scratch** (pas de warm-start depuis checkpoint 10k) sur 40k train.
- Mêmes hyperparams, sauf epochs max = 15 (plus de données compense).
- Checkpoints → `artifacts/scale_50k/{top1,top2}.safetensors`.
- Éval test set → winner final sur KL_total test.

### Phase 3 — Packaging gagnant (GrosMac, ~30 min)

- Export weights du gagnant en `.safetensors` + forward pure-NumPy via `NumpyExporter`.
- **Sanity check** : forward NumPy vs forward JAX, max diff < 1e-5 sur 100 queries test.
- Version nommée `v0.3-text-{B|C|D}-50k.safetensors`, promue en `v0.3.safetensors` (remplace `v0.2-d128.safetensors` dans `KikiFlowBridge` config).

### Kill-switch supplémentaire (R3 mitigation)

Si à la fin de Phase 1 l'écart relatif entre rang-1 et rang-3 en `KL_total(test)` est **< 15 %**, promouvoir les 3 archis au scale 50k (coût +1 run, ~2h) au lieu de 2 — le choix Top-2 serait trop bruité.

## 6. Métriques & évaluation

### Primaire (sélection gagnant)

**KL-total test** : `(1/|S|) · Σ_{s ∈ S} KL(ρ_target_s || ρ_pred_s)` moyenné sur test set, avec S = {phono, sémantique, lexical, syntaxique}. Plus bas = mieux.

### Secondaires (figure ablation paper)

- **KL-par-species** : 4 valeurs séparées → barres empilées par archi.
- **MAPE_Δ** : `mean |Δ_pred - Δ_target|_1 / (|Δ_target|_1 + ε)`, métrique rétro-compatible avec v0.2.
- **Routing hit@5** : blend `0.9 · base + 0.1 · bridge_pred` → top-5 stacks prédits vs top-5 oracle, intersection non vide.
- **Latency inference** : ms/query, P50/P95, mesuré JAX puis NumPy.
- **Model size** : nombre de params + taille `.safetensors` en KB.

### Scaling curves (figure paper)

- Axe x : {10k, 50k} (2 points, log-spaced).
- Axe y : KL_total test.
- 3 courbes pointillées {B, C, D} à 10k.
- 2 courbes pleines {Top-1, Top-2} continuant à 50k.
- Barre verticale horizontale : baseline v0.2-d128 (no text) comme référence.

### Loss d'entraînement

```
L_train = MSE(Δ_pred, Δ_target) + λ · KL_total(ρ_pred, ρ_target),  λ = 0.5
```

Rationale : MSE garde compatibilité avec v0.2 et régularise vers la dynamique flux globale. KL par species aligne les distributions species-specific (ce qui est l'objectif paper).

## 7. Machine orchestration

| Machine | Rôle | Runtime | Commande indicative |
|---------|------|---------|---------------------|
| **Studio (M3 Ultra 512GB)** | Phase 0 smoke JKO, génération paires JKO (10k puis +40k incrémental) | MLX, python 3.14 uv | `uv run python -m track3_deploy.jko_oracle_runner --corpus corpus_hybrid_v1.jsonl --out pairs_v1/ --cache-dir .jko_cache/` |
| **KXKM (RTX 4090)** | Génération synthétique Qwen (part D), training 3 encodeurs pilote + Top-2 scale | JAX+CUDA via venv `~/venv`, tunnel autossh réutilisé | `python -m track3_deploy.sweep --phase pilot10k --archs B,C,D` puis `--phase scale50k --archs top2` |
| **GrosMac (M5)** | Éval, figures matplotlib, packaging NumPy gagnant | NumPy, matplotlib | `python -m track3_deploy.eval.kl_species --checkpoints pilot_10k/ scale_50k/` |

### Data transport

- **Corpus hybride** : assemblé sur GrosMac (source lightweight) → rsync vers Studio + KXKM.
- **Paires JKO** : Studio → KXKM via rsync direct (Tailscale `kxkm-ai`), ~1 GB à 50k. Pas via GrosMac (éviter goulot RAM).
- **Checkpoints** : KXKM → GrosMac (rsync). Logs W&B ou Langfuse sur Tower (réutilise infra existante).
- **JKO cache** : `.safetensors` indexé par `sha256(query)` → pas de re-calc entre phases.

### Coexistence Studio

Studio actuellement occupé par SFT 35B Opus + distill 35B Opus parallèle (ETA ~2h au moment du brainstorming). JKO oracle tourne aussi en MLX → **gating mémoire obligatoire** :

- Script `scripts/studio_gate.sh` : lance JKO seulement si RAM libre > 80 GB.
- Sinon, fallback : laisse Studio finir, queue JKO après (décale timeline de ~2h, acceptable).

### Go/no-go KXKM

**Go explicite reçu pour ce training** (brainstorming 2026-04-19). Pas de re-demande requise pour les phases 1-2 de ce sprint. La règle générale (`feedback_no_launch_kxkm_without_ask`) reste active pour les prochains projets.

### Reproductibilité

- Seeds fixes : `torch=0, jax=0, numpy=0, PYTHONHASHSEED=0`.
- Versions figées dans `uv.lock` + `pip freeze` côté venv KXKM.
- Git commit tag avant chaque phase (`phase0-smoke`, `phase1-pilot10k`, `phase2-scale50k`).
- Tous artefacts horodatés ISO8601 + checksum SHA256 dans `artifacts/MANIFEST.sha256`.

## 8. Risques & questions ouvertes

| ID | Risque | Impact | Mitigation |
|----|--------|--------|------------|
| **R1** | JKO oracle n'expose pas `rho_by_species` séparés | **Bloquant** (KL-par-species infaisable) | Phase 0 smoke test valide API avant Phase 1. Si manquant : patch `oracle()` pour retourner dict `{species: rho}`. |
| **R2** | Biais couverture species dans D (Qwen) | Skew KL-par-species en faveur de sémantique | Après génération, mesurer distribution KL-target-moyen par species. Si déséquilibre > 2× : sur-sampler species sous-représentées depuis B. |
| **R3** | Flip d'ordre archi entre 10k et 50k (tiny-TF under-trained à 10k) | Mauvaise sélection Top-2 | Top-2 déjà choisi (vs Top-1). Kill-switch additionnel : si écart rang-1 vs rang-3 < 15 %, promouvoir les 3 au scale. |
| **R4** | Leakage corpus dedup incomplet | Inflation artificielle metrics test | Audit manuel 200 paires (val+test) sur 50k ; si > 2 % quasi-identiques, baisser seuil dedup à 0.88 et re-run. |
| **R5** | Coût Qwen 20k queries via tunnel (~3h KXKM occupé) | Conflit avec training parallèle | Séquencer : générer synthétique **avant** training, pas en parallèle. Cache réponses Qwen. |
| **R6** | Deadline paper non définie | Priorité timeline floue | Placeholder 2 semaines. À caler avec cycle de soumission visé (NeurIPS deadline May 2026 ? ICLR 2027 ?). |
| **R7** | λ = 0.5 pas swepé (future work) | Possible sous-optimalité loss balance | Documenté ; sweep λ ∈ {0.1, 0.5, 1.0} reporté à v0.4 si gains marginaux attendus. |
| **R8** | Studio mémoire saturée pendant SFT/distill en cours | Phase 0/1 JKO queue | Gating mémoire 80 GB (§7). Timeline buffer 2h. |

### Questions ouvertes à trancher avant Phase 1

1. **R6 (deadline)** : cycle paper visé, impact sur timeline 1 semaine vs 2 semaines.
2. **R7 (λ sweep)** : inclure dans v1 ou différer à v0.4 ?
3. **Calibration curve / reliability diagram** par species : ajouter aux métriques secondaires ?
4. **Eval humaine** sur 50 queries : inclure (coût ~2h manuel) ou omettre (paper FR ML est accepté sans en général) ?

## 9. Tests & validation

### Unit tests (obligatoires avant Phase 1)

| Test | Module | Assertion principale |
|------|--------|----------------------|
| `test_corpus_builder.py` | `data/corpus_builder.py` | Dedup exact + embedding ; stratification splits respecte ratios ± 5 %; cross-source dedup |
| `test_synth_qwen.py` | `data/synth_qwen.py` | Mock tunnel HTTP ; parsing réponse Qwen (une query/ligne) ; species tagging correct |
| `test_jko_oracle.py` | `jko_oracle_runner.py` | Retourne `(state_pre, state_post, rho_by_species)` avec shape `(4, 32)` |
| `test_encoders_shapes.py` | `encoders/*.py` | Chacun des 3 encodeurs produit 384-dim sur inputs {court, long, unicode, vide} |
| `test_joint_trainer.py` | `surrogate_trainer_v3.py` | Overfit-1-batch : loss décroît de > 50 % en 100 steps |
| `test_eval_kl.py` | `eval/kl_species.py` | KL-par-species = 0 sur (pred = target) ; > 0 sinon ; symétrie numerique |
| `test_numpy_export.py` | `export/to_numpy.py` | Max diff forward JAX vs NumPy < 1e-5 sur 100 queries test |

### Integration test (avant Phase 1)

Pipeline end-to-end sur 100 queries : `corpus → JKO → train 5 epochs → eval → export`. Doit tourner en **< 5 min sur GrosMac**. Tag `integration-v1` si ça passe.

### Reproductibilité

- Re-run Phase 1 avec même seed → `KL_total(test)` match à ± 0.001.
- Checksum SHA256 sur tous artefacts finaux.
- CI (`.github/workflows/`) : run unit tests sur push, pas d'intégration (trop lourd).

## 10. Deliverables & timeline

### Livrables

1. **Code** : nouveau module `kiki_flow_core/track3_deploy/` sous-dirs `encoders/`, `data/`, `eval/`, `export/` + fichiers top-level `sweep.py`, `jko_oracle_runner.py`, `surrogate_trainer_v3.py`.
2. **Weights** :
   - `v0.3-text-B-10k.safetensors`, `v0.3-text-C-10k.safetensors`, `v0.3-text-D-10k.safetensors` (pilote).
   - `v0.3-text-{top1,top2}-50k.safetensors` (scale).
   - `v0.3.safetensors` (gagnant final, promu, pure-NumPy).
3. **Figure paper** : `paper/figures/text_surrogate_ablation.{pdf,png}` (2 panels : species breakdown + scaling curves).
4. **Section paper** : `paper/sections/4_text_native_bridge.tex` pour paper v0.3.
5. **Design doc** : ce fichier, `docs/superpowers/specs/2026-04-19-text-bridge-surrogate-design.md`.
6. **PR micro-kiki** : swap `v0.2-d128` → `v0.3` dans `KikiFlowBridge` config (dans le repo `micro-kiki`, pas `kiki-flow-research`).
7. **Corpus artefact** : `corpus_hybrid_v1.jsonl` + SHA256, stocké dans `data/processed/` (gitignored, upload HF dataset optionnel).

### Timeline (best-case, aucun blocker)

| Jour | Étape | Machine |
|------|-------|---------|
| J+0 (2026-04-19) | Spec approuvé, plan écrit, tests unitaires squelette | GrosMac |
| J+1 (2026-04-20) | Corpus B+C assemblé, dedup pipeline validé | GrosMac |
| J+2 (2026-04-21) | Synthétique D généré via Qwen (~3h) + dedup final cross-source | KXKM |
| J+3 (2026-04-22) | Phase 0 smoke JKO (Studio) + Phase 1 pilote 10k (3 archis, KXKM) | Studio + KXKM |
| J+4 (2026-04-23) | Éval pilote, ranking Top-2, décision kill-switch R3 | GrosMac |
| J+5-6 (2026-04-24/25) | Phase 2 scale 50k sur Top-2 (+40k paires JKO incrémentales + training) | Studio + KXKM |
| J+7 (2026-04-26) | Éval scale, figure paper, packaging gagnant NumPy | GrosMac |
| J+8 (2026-04-27) | PR micro-kiki v0.3, section paper TeX, tag release | GrosMac |

**Soit ~1 semaine wall-clock** si rien ne casse. Avec marges de sécurité pour R1-R5 : **~2 semaines réaliste** (fin 2026-05-03).

## 11. Section paper « Text-Native Bridge Surrogate »

### Positionnement

Section 4.x du paper v0.3 (après Section 4 « Bridge Surrogate » de v0.2). Titre candidat : *« End-to-End Text-Conditioned Bridge Surrogate: Architecture Ablation and Scaling »*.

### Narratif prévu (~800 mots + figures)

1. **Motivation** (~150 mots) : limites du surrogate v0.2 entraîné sur paires JKO synthétiques sans texte, fallback SHA256 non-sémantique. Question de recherche : quelle architecture d'encodeur minimise la divergence entre distributions de flux prédites et oracle, sous contrainte de déployabilité ?
2. **Method** (~300 mots) : corpus hybride B+C+D, 3 architectures encodeurs, JKO oracle exposant `rho_by_species`, joint training encoder+bridge, loss combinée MSE + λ·KL.
3. **Protocol** (~100 mots) : pilot 10k → rank → scale Top-2 50k, seeds fixes, reproducibility statement.
4. **Results** (~200 mots) : lecture Figure 4.x + Table 4.x, identification du gagnant, species-specific behavior, observation sur scaling (3→2 cut).
5. **Discussion** (~50 mots) : limites (biais Qwen, FR only, λ non swepé), future work (multilingue, λ sweep v0.4, calibration).

### Métriques formalisées

$$
\mathrm{KL}_{\text{species}}(q, p) = \sum_{i=1}^{32} q_i \log \frac{q_i + \varepsilon}{p_i + \varepsilon}, \quad \varepsilon = 10^{-8}
$$

$$
\mathrm{KL}_{\text{total}} = \frac{1}{|S|} \sum_{s \in S} \mathrm{KL}_{\text{species}}(\rho^{\text{target}}_s, \rho^{\text{pred}}_s), \quad S = \{\text{phono}, \text{sém}, \text{lex}, \text{synt}\}
$$

$$
\mathrm{MAPE}_\Delta = \frac{1}{N} \sum_{n=1}^{N} \frac{\|\Delta_n^{\text{pred}} - \Delta_n^{\text{target}}\|_1}{\|\Delta_n^{\text{target}}\|_1 + \varepsilon}
$$

$$
\mathrm{hit@5}_{\text{routing}} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\!\left[ \mathrm{top5}(0.9 \cdot \mathbf{b}_n + 0.1 \cdot \hat{\mathbf{r}}_n) \cap \mathrm{top5}(\mathbf{b}_n + \mathbf{r}_n^{\text{oracle}}) \neq \emptyset \right]
$$

Notation : $\mathbf{b}_n \in \mathbb{R}^{32}$ scores de base par stack (avant blend), $\hat{\mathbf{r}}_n$ advisory prédit par le bridge surrogate, $\mathbf{r}_n^{\text{oracle}}$ advisory oracle (calculé offline via JKO complet, sans blend). Intersection non vide = le top-5 blend approxime le top-5 oracle sur au moins un stack.

$$
\mathcal{L}_{\text{train}} = \mathrm{MSE}(\Delta^{\text{pred}}, \Delta^{\text{target}}) + \lambda \cdot \mathrm{KL}_{\text{total}}, \quad \lambda = 0.5
$$

### Table 4.x — Encoder Ablation on Text-Conditioned Bridge Surrogate

| Archi | Params | KL-total ↓ | KL-phono ↓ | KL-sém ↓ | KL-lex ↓ | KL-synt ↓ | MAPE_Δ ↓ | hit@5 ↑ | Latency (ms) | Size (KB) |
|-------|-------:|-----------:|-----------:|---------:|---------:|----------:|---------:|--------:|-------------:|----------:|
| v0.2 baseline (no text) | 360K | *ref* | *ref* | *ref* | *ref* | *ref* | *ref* | *ref* | 2.1 | 920 |
| B. MiniLM-distilled (10k) | 2.1M | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX | X.XX | X.XX | X.X | XXX |
| C. Hash+MLP (10k) | 520K | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX | X.XX | X.XX | X.X | XXX |
| D. Tiny-Transformer (10k) | 8.3M | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX | X.XX | X.XX | X.X | XXX |
| **Top-1 (50k)** | X.XM | **X.XXX** | **X.XXX** | **X.XXX** | **X.XXX** | **X.XXX** | **X.XX** | **X.XX** | **X.X** | **XXX** |
| **Top-2 (50k)** | X.XM | X.XXX | X.XXX | X.XXX | X.XXX | X.XXX | X.XX | X.XX | X.X | XXX |

Caption : *« Encoder architecture ablation on text-conditioned bridge surrogate. All metrics on held-out test set (10% of hybrid corpus, N ≈ 5000). Best per column in bold. v0.2 baseline is bridge trained on synthetic JKO pairs without text conditioning. »*

### Figure 4.x — 2 panels

**Panel (a) — Species breakdown.** Barres empilées KL-par-species à 50k pour Top-1, Top-2, et baseline v0.2.

- Axe y : KL (log scale éventuellement si étalement > 2 décades).
- Axe x : {v0.2, Top-1, Top-2}.
- Couleurs distinctes par species (palette `viridis` ou `cividis` pour accessibilité).
- Caption : *« Species-specific KL divergence on test set. Lower is better. Stacked bars decompose KL_total into phono, semantic, lexical, syntactic contributions. »*

**Panel (b) — Scaling behavior.** Scaling curves.

- Axe x : corpus size ∈ {10k, 50k} (log-spaced, 2 ticks).
- Axe y : KL_total test.
- 3 courbes pointillées {B, C, D} à 10k uniquement (points isolés + marqueur).
- 2 courbes pleines {Top-1, Top-2} continuant à 50k (ligne + marqueur).
- Barre horizontale pointillée : baseline v0.2 (no text).
- Caption : *« Scaling behavior on hybrid corpus. 3 architectures evaluated at pilot (10k); top-2 continue to scale (50k). Baseline v0.2 (no text conditioning) shown as horizontal reference. »*

### Baselines & références supplémentaires

- **v0.2 hand-crafted** : MAPE_Δ et KL_total mesurés rétroactivement sur test set v1 pour remplir la ligne « ref ».
- **Random encoder** (untrained Gaussian init) : KL ceiling, mesuré une fois (seed=0) → cité en texte *« all three architectures beat random by > X× on KL_total »*.
- **FastText FR pretrained** (optionnel si temps) : comparaison additionnelle comme baseline pré-entraînée non-distillée.

### Hyperparamètres (appendix A.x)

- lr, batch, epochs, warmup, λ par archi.
- Corpus proportions, dedup threshold, seeds.
- JKO oracle version + commit hash.
- Hardware used + wall-clock per phase.

### Reproducibility statement (obligatoire)

- **Code** : repo `electron-rare/kiki-flow-research` au commit `<hash>` (tag `v0.3-paper`).
- **Corpus** : SHA256 du fichier `corpus_hybrid_v1.jsonl` + instructions recompilation. B+C publiquement re-dérivables depuis sources ouvertes ; D regenerable via prompts Qwen fournis en appendix.
- **Compute** : ~6h KXKM 4090 + ~4h Studio MLX + ~1h GrosMac eval (total wall-clock). Single-run reproductibilité ± 0.001 en KL_total avec seeds fixes.

---

## Appendix — Brainstorming decisions trace

Pour audit future : les 10 questions de brainstorming et les choix utilisateur (2026-04-19).

| Q | Sujet | Choix retenu |
|---|-------|--------------|
| 1 | Périmètre | **C** — End-to-end (nouveau corpus + nouvel encodeur + surrogate) |
| 2 | Corpus | **B+C+D** — psycholinguistique + généraliste FR + synthétique LLM |
| 3 | Encodeur | **B+C+D** — distillé MiniLM + hash+MLP + tiny-transformer |
| 4 | Nature B+C+D | **A** — sweep ablatif (3 archis indépendantes, figure paper) |
| 5 | Volume | **D** — incrémental : 10k pilote → 50k scale sur gagnant(s) |
| 6 | Métriques | **D** — paper-centric : KL-par-species + ablation + scaling curve |
| 7 | Machine | **D** — pipeline distribuée Studio+KXKM+GrosMac |
| 8 | Déploiement | **C** — flex pendant training, export NumPy pour gagnant seulement |
| 9 | LLM synthétique | **A** — Qwen3.5-35B local via tunnel KXKM (0 cost) |
| 10 | Orchestration | **②** — Top-2 gating (10k→50k sur 2 finalistes) |
| extra | Paper | Section formalisée avec métriques + tableaux + figures |

**Go KXKM-AI** : accordé explicitement pour ce sprint (2026-04-19). Règle générale de demande préalable reste active pour les projets suivants.
