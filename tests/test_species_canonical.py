import numpy as np

from kiki_flow_core.species.canonical_species import CanonicalSpecies


def test_ortho_species_names_in_canonical_order():
    s = CanonicalSpecies()
    assert s.species_names() == ["phono", "lex", "syntax", "sem"]


def test_ortho_coupling_matrix_shape():
    s = CanonicalSpecies()
    j = s.coupling_matrix()
    assert j.shape == (4, 4)
    assert np.isfinite(j).all()


def test_ortho_coupling_matrix_non_negative():
    s = CanonicalSpecies()
    j = s.coupling_matrix()
    assert (j >= 0).all(), "Spreading activation strengths should be non-negative"


def test_ortho_coupling_strongest_diag_or_known_links():
    s = CanonicalSpecies()
    j = s.coupling_matrix()
    idx = {n: i for i, n in enumerate(s.species_names())}
    assert j[idx["lex"], idx["sem"]] >= 0.5, "sem->lex spreading activation should dominate"  # noqa: PLR2004


def test_ortho_n_species_property():
    s = CanonicalSpecies()
    assert s.n_species == 4  # noqa: PLR2004
