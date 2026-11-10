"""Structure of the coefficient tables (Oosterveld & van Oossanen 1975)."""
from wageningen_bseries import KQ_TERMS, KT_TERMS


def test_term_counts():
    assert len(KT_TERMS) == 39
    assert len(KQ_TERMS) == 47


def test_exponents_are_whole_numbers_in_published_sets():
    for table in (KT_TERMS, KQ_TERMS):
        for c, s, t, u, v in table:
            assert isinstance(c, float)
            assert s in (0, 1, 2, 3)  # J: at most cubic -> K_T(J), K_Q(J) are cubics
            assert t in (0, 1, 2, 3, 6)  # P/D
            assert u in (0, 1, 2)  # AE/A0
            assert v in (0, 1, 2)  # Z


def test_no_repeated_monomials():
    for table in (KT_TERMS, KQ_TERMS):
        exps = [term[1:] for term in table]
        assert len(set(exps)) == len(exps)


def test_first_and_last_terms():
    # Spot checks against the printed tables.
    assert KT_TERMS[0] == (0.00880496, 0, 0, 0, 0)
    assert KT_TERMS[-1] == (-0.00146564, 0, 3, 2, 2)
    assert KQ_TERMS[0] == (0.00379368, 0, 0, 0, 0)
    assert KQ_TERMS[-1] == (0.0000554194, 1, 6, 2, 2)
