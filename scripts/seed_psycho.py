"""Generate canonical psycholinguistic seeds for B-source via Qwen tunnel.

Overwrites data/raw/psycho/{species}.txt with ~2500 lines per species.
Uses stricter species-specific prompts than synth_qwen (source D) to target
canonical psycholinguistic examples (not just "species-aware generic").

Usage: PYTHONPATH=. uv run python scripts/seed_psycho.py [--n-per-species 2500]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

QWEN_URL = "http://localhost:18000/v1/chat/completions"
MODEL = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"
SPECIES_SHORT = ("phono", "sem", "lex", "syntax")

# Stricter, canonical prompts — shorter queries, one trait strongly exemplified.
CANONICAL_PROMPTS: dict[str, str] = {
    "phono": (
        "Génère 20 courtes phrases françaises (5 à 12 mots) où le trait "
        "phonologique est saillant et canonique : virelangues, assonances "
        "serrées, paires minimales (chat/rat, ver/vert), homophones ambigus "
        "(sans/sang/cent), suites consonantiques difficiles (strict, spleen). "
        "Une phrase par ligne, pas de numéro, pas de tiret, pas de guillemets."
    ),
    "sem": (
        "Génère 20 courtes phrases françaises (6 à 14 mots) qui exigent une "
        "désambiguïsation sémantique canonique : polysémie contextuelle "
        "(banque, lampe, voler), garden-path sémantique, antonymie vs "
        "synonymie fine (chagrin/tristesse), métonymie (boire un verre). "
        "Une phrase par ligne, pas de numéro, pas de tiret."
    ),
    "lex": (
        "Génère 20 courtes phrases françaises (7 à 14 mots) avec au moins "
        "un mot rare ou technique canonique : vocabulaire scientifique, "
        "néologismes récents, archaïsmes littéraires, termes spécialisés "
        "(photolyse, palimpseste, zeugme, syllogisme). Une phrase par "
        "ligne, pas de numéro, pas de tiret."
    ),
    "syntax": (
        "Génère 20 courtes phrases françaises (10 à 22 mots) à structure "
        "syntaxique canoniquement complexe : subordonnées relatives "
        "emboîtées, ambiguïté d'attachement prépositionnel (le livre de "
        "l'ami que tu aimes), coordination asymétrique, dépendance longue "
        "sujet-verbe. Une phrase par ligne, pas de numéro, pas de tiret."
    ),
}

logger = logging.getLogger(__name__)


def call_qwen(prompt: str, client: httpx.Client, attempt_max: int = 3) -> list[str]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(1, attempt_max + 1):
        try:
            r = client.post(QWEN_URL, json=payload, timeout=120.0)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            return [
                line.strip().lstrip("0123456789.-) ").strip()
                for line in content.splitlines()
                if len(line.strip()) >= 4  # noqa: PLR2004
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("call failed attempt %d: %s", attempt, e)
            time.sleep(2**attempt)
    return []


def generate_for_species(species: str, n_target: int, out_path: Path) -> int:
    prompt = CANONICAL_PROMPTS[species]
    seen: set[str] = set()
    lines: list[str] = []
    client = httpx.Client()
    try:
        stall = 0
        while len(lines) < n_target and stall < 10:  # noqa: PLR2004
            batch = call_qwen(prompt, client)
            new = [q for q in batch if q and q.lower() not in seen]
            for q in new:
                seen.add(q.lower())
                lines.append(q)
            logger.info(
                "species=%s got batch=%d new=%d total=%d/%d",
                species,
                len(batch),
                len(new),
                len(lines),
                n_target,
            )
            stall = stall + 1 if len(new) == 0 else 0
    finally:
        client.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines[:n_target]) + "\n", encoding="utf-8")
    return len(lines[:n_target])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-species", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/psycho"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    totals: dict[str, int] = {}
    for species in SPECIES_SHORT:
        out = args.out_dir / f"{species}.txt"
        t0 = time.time()
        n = generate_for_species(species, args.n_per_species, out)
        elapsed = time.time() - t0
        totals[species] = n
        logger.info("species=%s wrote %d lines to %s (%.0f s)", species, n, out, elapsed)
    logger.info("all totals: %s", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
