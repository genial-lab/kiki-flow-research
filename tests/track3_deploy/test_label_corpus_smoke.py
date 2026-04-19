"""Smoke test for label_corpus.py CLI on 3 fake queries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spacy = pytest.importorskip("spacy")

_N_QUERIES = 3
_N_SPECIES = 4
_N_BINS = 32


def _load_label_corpus_main():  # type: ignore[return]
    spec_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "label_corpus.py"
    spec = importlib.util.spec_from_file_location("label_corpus", spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {spec_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.main


def test_label_corpus_smoke(tmp_path: Path) -> None:
    """Run label_corpus.main on 3 queries, verify NPZ written with correct shape."""
    main = _load_label_corpus_main()

    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w") as fh:
        for q in (
            "Bonjour le monde.",
            "Ceci est un test.",
            "Une autre query française.",
        ):
            fh.write(json.dumps({"text": q, "source": "B", "species": "phono"}) + "\n")
    output = tmp_path / "labels.npz"
    rc = main(["--corpus", str(corpus), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    data = np.load(output, allow_pickle=True)
    assert len(data.files) == _N_QUERIES
    for h in data.files:
        arr = data[h]
        expected = (_N_SPECIES, _N_BINS)
        assert arr.shape == expected, f"expected {expected}, got {arr.shape}"
        # each species row sums to 1 (simplex)
        sums = arr.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-4), f"simplex rows not normalised: {sums}"
