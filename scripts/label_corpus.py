"""CLI: read JSONL corpus, run HeuristicLabeler, save per-query labels to NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

SPECIES_CANONICAL = ("phono:code", "sem:code", "lex:code", "syntax:code")
LOG_EVERY = 500
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexique", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from kiki_flow_core.track3_deploy.data.heuristic_labeler import (  # noqa: PLC0415
        HeuristicLabeler,
    )

    labeler = HeuristicLabeler(lexique_csv=args.lexique)

    out: dict[str, np.ndarray] = {}
    n = 0
    with args.corpus.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            q = entry["text"]
            h = hashlib.sha256(q.encode("utf-8")).hexdigest()
            labels = labeler.label(q)
            stacked = np.stack([labels[sp] for sp in SPECIES_CANONICAL]).astype(np.float32)
            out[h] = stacked
            n += 1
            if n % LOG_EVERY == 0:
                logger.info("labeled %d queries", n)
    np.savez_compressed(args.output, **out)
    logger.info("wrote %d labels to %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
