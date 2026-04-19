"""Precompute fixed encoder embeddings for each corpus query (used as g_JEPA target)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from kiki_flow_core.track3_deploy.encoders.hash_mlp import EncoderC_HashMLP

_BATCH = 256
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    enc = EncoderC_HashMLP(seed=args.seed)
    out: dict[str, np.ndarray] = {}
    texts_buf: list[str] = []
    hashes_buf: list[str] = []

    def _flush() -> None:
        if texts_buf:
            embs = enc.encode(texts_buf)
            for h, e in zip(hashes_buf, embs, strict=True):
                out[h] = e
            texts_buf.clear()
            hashes_buf.clear()

    n = 0
    with args.corpus.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            q = entry["text"]
            hashes_buf.append(hashlib.sha256(q.encode("utf-8")).hexdigest())
            texts_buf.append(q)
            n += 1
            if len(texts_buf) >= _BATCH:
                _flush()
        _flush()
    np.savez_compressed(args.output, **out)
    logger.info("wrote %d embeddings to %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
