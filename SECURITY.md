# Security policy — kiki-flow-research

kiki-flow-research is a **research code base** and the upstream
numerical engine for the dreamOfkiki research program. Correctness
and reproducibility are our top priorities ; security is handled on
a best-effort basis.

## Scope

This policy applies to :

- The code in this repository (`genial-lab/kiki-flow-research`)
- The numerical routines used downstream by `dream-of-kiki`,
  `micro-kiki`, and `nerve-wml`
- The documentation in `docs/`

It does *not* apply to dependencies (NumPy, MLX, PyTorch, etc.) ;
please report those upstream.

## Reporting a vulnerability

If you discover a security issue that could compromise :

- the bit-exact behaviour of the Wasserstein gradient flow solver
- the determinism of divergence estimators under fixed seeds
- the integrity of any pre-computed species-ontology artifact
- silent numerical errors that would invalidate downstream reproducibility

please report it **privately** via one of these channels :

1. Email : `clement@saillant.cc` — subject starting with `[SECURITY]`
2. GitHub Private Vulnerability Reporting :
   https://github.com/genial-lab/kiki-flow-research/security/advisories/new

Please include :

- a description of the issue
- steps to reproduce (ideally with a deterministic seed)
- the commit SHA of the affected run if applicable
- suggested mitigation or patch if you have one

We aim to acknowledge reports within **5 business days** and publish
a fix within **30 days** for critical issues. Reporters will be
credited unless they prefer otherwise.

## Out of scope

- General quality-of-numerical-results questions : please open a
  regular GitHub issue.
- Dependency vulnerabilities : report upstream.
- Questions about licensing : see `LICENSE`.

## Threat model

The expected threat is *inadvertent numerical or reproducibility
bugs* ; we do not defend against a malicious contributor with
write access. Review happens via pull-request discipline and CI.
