import numpy as np

from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder


def test_encoder_returns_fixed_dim_vector():
    enc = QueryEncoder(use_stub=True)
    vec = enc.encode("hello world")
    assert vec.shape == (384,)  # noqa: PLR2004
    assert np.isfinite(vec).all()


def test_encoder_deterministic_on_same_input():
    enc = QueryEncoder(use_stub=True)
    v1 = enc.encode("hello")
    v2 = enc.encode("hello")
    np.testing.assert_array_equal(v1, v2)


def test_encoder_lru_cache_hit():
    enc = QueryEncoder(use_stub=True, cache_size=4)
    _ = enc.encode("a")
    _ = enc.encode("a")
    stats = enc.cache_stats()
    assert stats["hits"] >= 1
