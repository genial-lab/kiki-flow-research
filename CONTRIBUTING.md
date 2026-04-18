# Contributing

Thanks for looking at `kiki-flow-research`. This is active research code
that accompanies an in-progress paper; contributions are welcome but
please read the notes below before opening a pull request.

## Scope

The repository covers the core formalism (`kiki_flow_core/`), three
implementation tracks (`track1_perf`, `track2_paper`, `track3_deploy`),
reproducibility scripts, and a LaTeX paper. Contributions that fit
naturally into one of these components are easy to review. Contributions
that change the formalism, the species decomposition, or the core
abstractions require a discussion first (open an issue).

## Ground rules

1. **Keep the test suite green.** Run `uv run pytest` locally. New
   modules need tests; new figures need a regeneration script in
   `scripts/`.
2. **Strict `ruff` and `mypy`.** The CI enforces both on `kiki_flow_core/`.
   Pre-commit hooks auto-format on commit.
3. **Commit message format.** Subject ≤ 50 characters, conventional
   prefix (`feat:`, `fix:`, `docs:`, `perf:`, `test:`, etc.), body lines
   ≤ 72 characters. A hook rejects non-conforming messages.
4. **Honest reporting.** If a change produces a mixed-sign or negative
   finding, report it in the paper body alongside the positive results.
   `paper/main.tex` already has three such paragraphs as reference.
5. **Advisory-only integration.** Hooks into micro-kiki or any other
   LLM framework must never be able to break that system. Use
   try/except, default-off env flags, and swallow exceptions on the
   hot path.

## Development workflow

```bash
git clone https://github.com/electron-rare/kiki-flow-research.git
cd kiki-flow-research
uv sync --all-extras
uv run pre-commit install

# make changes...

uv run ruff check --fix .
uv run mypy kiki_flow_core
uv run pytest

git add -A
git commit -m "feat(scope): short description"
git push -u origin feat/your-branch
gh pr create
```

## What we are actively looking for

- A real continual-learning benchmark that replaces the distributional
  proxy in `scripts/cl_benchmark.py`.
- A measured sweep of `(α, β, γ, λ_T)` denser than the current 27
  configurations in `scripts/hyperparam_sweep.py`.
- A native MLX Sinkhorn implementation that matches POT numerically
  (draft scaffold in `kiki_flow_core/track2_paper/mlx_wasserstein.py`).
- A clean integration of the surrogate into an actual micro-kiki
  inference pathway (the four integration dependencies are tracked in
  `docs/superpowers/integration-notes.md`).
- Any reviewer feedback on the paper draft at tag `paper-v0.4-draft`.

## What we are not looking for

- Generic refactors without a behavior change.
- Dependency upgrades for their own sake.
- Style-only churn (`ruff-format` already handles formatting).
- New tracks beyond the three current ones without a design discussion.

## Reporting issues

Open a GitHub issue with:
- A short description of the problem or proposal.
- The minimum reproducer (or a link to the test that surfaces it).
- The environment (macOS version, Python version, `uv --version`).

## License

By contributing you agree that your contribution is licensed under the
repository's MIT License (see `LICENSE`).
