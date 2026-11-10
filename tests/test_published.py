"""Comparison with published values computed from the same 1975 polynomials.

Source: S. Helma (2020), "Surprising behaviour of the Wageningen B-screw series
polynomials", J. Mar. Sci. Eng. 8(3), 211, https://doi.org/10.3390/jmse8030211
(open access, CC BY 4.0).

* Table A3 works "Example 1" of G. Kuiper, *The Wageningen Propeller Series*
  (MARIN, 1992): a B4-55 propeller with known thrust, diameter and speed of
  advance, i.e. a fixed K_T / J**2. Helma lists Kuiper's iterations and the
  exact optimum computed from the Oosterveld & van Oossanen (1975) polynomials.
* Section 3.1 gives operating points read from his efficiency diagrams, which
  he also computed from the 1975 polynomials. These are graphical readings,
  so the tolerances are those of reading a chart.

Tolerances are chosen from the number of digits printed, never tighter.
"""
import numpy as np
import pytest

import wageningen_bseries as wb

B4_55 = dict(EAR=0.55, Z=4)


def j_for_kt_over_j2(PD, EAR, Z, ratio):
    """Advance coefficient where K_T / J**2 equals ``ratio``.

    K_T(J) - ratio * J**2 = 0 is a cubic in J; take its smallest positive root.
    """
    c = wb.cubic_coefficients(PD, EAR, Z).kt.copy()
    c[2] -= ratio
    roots = np.roots(c[::-1])
    real = roots[np.isreal(roots)].real
    return real[real > 0].min()


def test_helma_table_a3_exact_solution():
    """Table A3, 'Exact solution': P/D 1.004, J 0.7007, KT 0.1823, KQ 0.03124, eta0 0.6509."""
    ratio = 0.371310
    pd = 1.004
    J = j_for_kt_over_j2(pd, ratio=ratio, **B4_55)
    ow = wb.open_water(J, pd, **B4_55)
    assert J == pytest.approx(0.7007, abs=0.5e-4)
    assert ow.kt == pytest.approx(0.1823, abs=0.5e-4)
    assert ow.kq == pytest.approx(0.03124, abs=0.5e-5)
    assert ow.eta0 == pytest.approx(0.6509, abs=0.5e-4)


def test_helma_table_a3_optimum_pitch():
    """The exact solution is the P/D of maximum eta0 at fixed K_T/J**2 = 0.371310."""
    ratio = 0.371310
    pds = np.arange(0.95, 1.05, 1e-4)
    etas = []
    for pd in pds:
        J = j_for_kt_over_j2(pd, ratio=ratio, **B4_55)
        etas.append(float(wb.eta0(J, pd, **B4_55)))
    best = int(np.argmax(etas))
    # Printed as 1.004 and 0.6509. The optimum is flat, so allow one more step in P/D.
    assert pds[best] == pytest.approx(1.004, abs=0.5e-3 + 1e-4)
    assert etas[best] == pytest.approx(0.6509, abs=0.5e-4)


@pytest.mark.parametrize(
    "pd, J, KT, KQ, eta0",
    [
        (0.95, 0.674, 0.168, 0.0278, 0.650),
        (1.00, 0.699, 0.181, 0.0310, 0.651),
        (1.05, 0.723, 0.194, 0.0344, 0.650),
    ],
)
def test_helma_table_a3_kuiper_iterations(pd, J, KT, KQ, eta0):
    """Kuiper's three iterations at K_T/J**2 = 0.3707 (value given in Helma's text).

    The values come from Kuiper's own program and are printed with three
    significant digits, so allow one unit in the last printed digit.
    """
    j = j_for_kt_over_j2(pd, ratio=0.3707, **B4_55)
    ow = wb.open_water(j, pd, **B4_55)
    assert j == pytest.approx(J, abs=1e-3)
    assert ow.kt == pytest.approx(KT, abs=1e-3)
    assert ow.kq == pytest.approx(KQ, abs=1e-4)
    assert ow.eta0 == pytest.approx(eta0, abs=1e-3)


@pytest.mark.parametrize(
    "name, Z, EAR, J, KT, KQ, PD, eta0",
    [
        ("B5-90", 5, 0.90, 0.65, 0.24, 0.041, 1.04, 0.60),
        ("B3-80, first optimum", 3, 0.80, 0.62, 0.20, 0.035, 1.00, 0.57),
        ("B3-80, second extremum", 3, 0.80, 0.74, 0.295, 0.062, 1.30, 0.56),
    ],
)
def test_helma_section_3_1_diagram_readings(name, Z, EAR, J, KT, KQ, PD, eta0):
    """Section 3.1: values 'read off' Helma's diagrams, printed as approximate."""
    ow = wb.open_water(J, PD, EAR, Z)
    assert ow.kt == pytest.approx(KT, abs=0.005)
    assert ow.kq == pytest.approx(KQ, abs=0.001)
    assert ow.eta0 == pytest.approx(eta0, abs=0.01)
