import pytest

from kiki_flow_core.species.hybrid_ortho_stacks import HybridSpecies


def test_hybrid_species_empty_stacks_raises():
    with pytest.raises(ValueError, match="non-empty"):
        HybridSpecies(stack_names=[])


def test_hybrid_species_identity_projection_gives_diagonal():
    s = HybridSpecies(stack_names=["a", "b", "c", "d"], projection_init="identity")
    p = s.projection_matrix()
    # Identity puts 1.0 on the first min(4, n_stacks) = 4 diagonal entries
    for i in range(4):
        assert p[i, i] == 1.0


def test_hybrid_species_unknown_projection_init_raises():
    with pytest.raises(ValueError, match="Unknown projection_init"):
        HybridSpecies(stack_names=["a"], projection_init="bogus")  # type: ignore[arg-type]


def test_hybrid_species_random_projection_rows_sum_to_one():
    s = HybridSpecies(stack_names=["a", "b", "c"], projection_init="random", seed=42)
    p = s.projection_matrix()
    for i in range(4):
        assert abs(p[i, :].sum() - 1.0) < 1e-6  # noqa: PLR2004
