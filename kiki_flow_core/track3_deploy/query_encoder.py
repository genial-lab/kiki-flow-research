"""MiniLM-based query encoder with LRU cache and a deterministic stub mode."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np


class QueryEncoder:
    """Encodes a query string to a 384-dim vector.

    In stub mode (``use_stub=True`` or if the real model is unavailable), the
    vector is a deterministic function of the query hash. This is for tests
    and for environments without sentence-transformers installed.
    """

    EMBED_DIM = 384

    def __init__(
        self,
        model_path: str | None = None,
        cache_size: int = 1024,
        use_stub: bool = False,
    ) -> None:
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self.use_stub = use_stub
        self._model: Any | None = None
        if not use_stub:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                self._model = SentenceTransformer(model_path or "all-MiniLM-L6-v2")
            except Exception:  # noqa: BLE001
                self.use_stub = True  # fall back silently

    def encode(self, query: str) -> np.ndarray:
        if query in self._cache:
            self._cache.move_to_end(query)
            self._hits += 1
            return self._cache[query]
        self._misses += 1
        vec = self._encode_raw(query)
        self._cache[query] = vec
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vec

    def _encode_raw(self, query: str) -> np.ndarray:
        if not self.use_stub and self._model is not None:
            out = self._model.encode(query, convert_to_numpy=True)
            if out.shape != (self.EMBED_DIM,):
                raise RuntimeError(f"Unexpected embed shape {out.shape}")
            result: np.ndarray = out.astype(np.float32)
            return result
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        raw = np.frombuffer(digest * (self.EMBED_DIM // 32 + 1), dtype=np.uint8)[: self.EMBED_DIM]
        return (raw.astype(np.float32) / 128.0) - 1.0

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}
