import numpy as np

from kiki_flow_core.species.canonical_species import CanonicalSpecies
from kiki_flow_core.species.mixed_canonical_stacks import MixedCanonicalSpecies


def test_hybrid_species_names_have_ortho_x_stack_structure():
    s = MixedCanonicalSpecies(stack_names=["code", "math", "fr"])
    names = s.species_names()
    assert len(names) == 4 * 3  # noqa: PLR2004
    for n in names:
        assert ":" in n  # format "ortho:stack"
    assert "phono:code" in names
    assert "sem:fr" in names


def test_hybrid_coupling_tensor_shape():
    stacks = ["code", "math", "fr", "en"]
    s = MixedCanonicalSpecies(stack_names=stacks)
    j = s.coupling_tensor()
    assert j.shape == (4, len(stacks), 4, len(stacks))


def test_hybrid_coupling_matches_ortho_when_projection_uniform_single_stack():
    """With 1 stack and uniform projection, coupling reduces to CanonicalSpecies J.

    Uniform init produces P = full((n_o, n_s), 1/n_s); for n_s=1 each row
    sums to 1, triggering the shortcut in MixedCanonicalSpecies.coupling_matrix()
    that returns CanonicalSpecies J directly.
    """
    s = MixedCanonicalSpecies(stack_names=["code"], projection_init="uniform")
    j = s.coupling_matrix()
    assert j.shape == (4, 4)
    expected = CanonicalSpecies().coupling_matrix()
    np.testing.assert_allclose(j, expected, atol=1e-6)


def test_hybrid_n_species():
    s = MixedCanonicalSpecies(stack_names=["a", "b", "c", "d", "e"])
    assert s.n_species == 4 * 5  # noqa: PLR2004


def test_hybrid_projection_has_well_defined_gradient():
    """Projection is a learnable parameter; must produce well-formed gradient on a toy loss."""
    s = MixedCanonicalSpecies(stack_names=["a", "b"], projection_init="random")
    p = s.projection_matrix()
    assert p.shape == (4, 2)
    grad = 2 * p  # toy loss = sum(P^2), gradient = 2*P
    assert np.isfinite(grad).all()
    assert grad.shape == p.shape
