"""Shared helpers: a plain term-by-term evaluation used as the reference."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from wageningen_bseries import KQ_TERMS, KT_TERMS

DATA = Path(__file__).parent / "data"
EPS = np.finfo(float).eps


def termwise_exact(terms, J, PD, EAR, Z):
    """Exact value of the polynomial (with the float coefficients) in rationals.

    Returns the exact sum rounded once to float, and the sum of the absolute
    values of the terms, which sets the scale of unavoidable rounding error.
    """
    J, PD, EAR, Z = (Fraction(float(x)) for x in (J, PD, EAR, Z))
    total = Fraction(0)
    scale = Fraction(0)
    for c, s, t, u, v in terms:
        term = Fraction(c) * J**s * PD**t * EAR**u * Z**v
        total += term
        scale += abs(term)
    return float(total), float(scale)


def assert_machine_precision(actual, exact, scale, ulps=8):
    """|actual - exact| <= ulps * eps * sum(|terms|), element-wise."""
    actual, exact, scale = (np.asarray(a, dtype=float) for a in (actual, exact, scale))
    err = np.abs(actual - exact)
    bound = ulps * EPS * np.maximum(scale, np.finfo(float).tiny)
    worst = np.max(err / bound)
    assert np.all(err <= bound), f"error up to {worst:.2f} x the bound of {ulps} ulps"


@pytest.fixture(scope="session")
def reference_rows():
    """Rows of tests/data/reference_termwise.csv as a float array."""
    return np.loadtxt(DATA / "reference_termwise.csv", delimiter=",", comments="#", skiprows=6)


__all__ = ["termwise_exact", "assert_machine_precision", "KT_TERMS", "KQ_TERMS"]
