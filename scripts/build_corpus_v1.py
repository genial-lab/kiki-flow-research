"""Build corpus_hybrid_v1.jsonl at target size (10k pilot or 50k scale).

Uses CorpusBuilder + SyntheticGenerator (T3, T4 from v0.3 sprint).
Source files expected under `data/raw/`:
  - psycho/{phono,sem,lex,syntax}.txt — one query per line per species
  - generalist.txt — generalist FR text (one query per line, 8-64 word lines)
  - Qwen tunnel must be reachable at localhost:18000 for the synthetic share.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from kiki_flow_core.track3_deploy.data.corpus_builder import (
    CorpusBuilder,
    CorpusEntry,
)

SPECIES_SHORT = ("phono", "sem", "lex", "syntax")
SOURCE_B_RATIO = 0.20
SOURCE_C_RATIO = 0.40
SOURCE_D_RATIO = 0.40
DEFAULT_DEDUP_THRESHOLD = 0.92
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
WIKI_LINE_MIN_TOKENS = 8
WIKI_LINE_MAX_TOKENS = 64

logger = logging.getLogger(__name__)


def load_psycholinguistic(n: int, seed: int = 0) -> list[CorpusEntry]:
    """Load B source — files under data/raw/psycho/{species}.txt."""
    src = Path("data/raw/psycho")
    items: list[CorpusEntry] = []
    rng = random.Random(seed)
    per_species = max(1, n // len(SPECIES_SHORT))
    for species in SPECIES_SHORT:
        path = src / f"{species}.txt"
        if not path.exists():
            logger.warning("psycho source missing: %s — skipping species", path)
            continue
        with path.open() as fh:
            lines = [line.strip() for line in fh if line.strip()]
        rng.shuffle(lines)
        items.extend(
            CorpusEntry(text=line, source="B", species=species) for line in lines[:per_species]
        )
    return items


def load_generalist(n: int, seed: int = 0) -> list[CorpusEntry]:
    """Load C source — generalist FR sample at data/raw/generalist.txt."""
    path = Path("data/raw/generalist.txt")
    if not path.exists():
        logger.warning("generalist source missing: %s", path)
        return []
    with path.open() as fh:
        lines = [
            line.strip()
            for line in fh
            if line.strip() and WIKI_LINE_MIN_TOKENS <= len(line.split()) <= WIKI_LINE_MAX_TOKENS
        ]
    rng = random.Random(seed)
    rng.shuffle(lines)
    species_cycle = SPECIES_SHORT
    return [
        CorpusEntry(
            text=line,
            source="C",
            species=species_cycle[i % len(species_cycle)],
        )
        for i, line in enumerate(lines[:n])
    ]


def generate_synthetic(n: int) -> list[CorpusEntry]:
    """Use SyntheticGenerator to generate N queries from Qwen tunnel."""
    from kiki_flow_core.track3_deploy.data.synth_qwen import (  # noqa: PLC0415
        SyntheticGenerator,
    )

    gen = SyntheticGenerator()
    out: list[CorpusEntry] = []
    per_species = max(1, n // len(SPECIES_SHORT))
    for species in SPECIES_SHORT:
        try:
            out.extend(gen.generate_tagged(species, per_species))
        except Exception as e:
            logger.warning("Qwen synth failed for species %s: %s", species, e)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build hybrid FR corpus for text-bridge sprint.")
    parser.add_argument("--size", type=int, required=True, help="Total target size (e.g. 10000)")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output dir for {train,val,test}.jsonl"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD)
    parser.add_argument(
        "--no-synth", action="store_true", help="Skip Qwen generation (use B+C only)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    n_b = int(args.size * SOURCE_B_RATIO)
    n_c = int(args.size * SOURCE_C_RATIO)
    n_d = args.size - n_b - n_c

    logger.info("targets: B=%d, C=%d, D=%d (total=%d)", n_b, n_c, n_d, args.size)
    b = load_psycholinguistic(n_b, seed=args.seed)
    c = load_generalist(n_c, seed=args.seed)
    d = [] if args.no_synth else generate_synthetic(n_d)
    all_entries = b + c + d
    logger.info("loaded: B=%d, C=%d, D=%d (raw total=%d)", len(b), len(c), len(d), len(all_entries))
    if not all_entries:
        logger.error("no corpus entries — aborting")
        return 1

    builder = CorpusBuilder(dedup_threshold=args.dedup_threshold)
    deduped = builder.dedup(all_entries)
    logger.info("after dedup: %d (was %d)", len(deduped), len(all_entries))
    splits = builder.split(deduped, ratios=DEFAULT_RATIOS, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, entries in splits.items():
        with (args.out / f"{name}.jsonl").open("w") as fh:
            for e in entries:
                fh.write(
                    json.dumps({"text": e.text, "source": e.source, "species": e.species}) + "\n"
                )
        logger.info("wrote %s split: %d entries", name, len(entries))
    test_hash = builder.freeze_hash(splits["test"])
    (args.out / "test.sha256").write_text(test_hash + "\n")
    logger.info("test split frozen, sha256=%s", test_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
