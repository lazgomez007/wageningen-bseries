"""K_T(J) and K_Q(J) as cubics for fixed P/D, AE/A0 and Z."""
import numpy as np
import pytest

import wageningen_bseries as wb
from conftest import KQ_TERMS, KT_TERMS, assert_machine_precision, termwise_exact


def test_shape_and_order():
    c = wb.cubic_coefficients(1.0, 0.55, 4)
    assert c.kt.shape == (4,) and c.kq.shape == (4,)
    # Constant term is K at J = 0.
    assert c.kt[0] == pytest.approx(float(wb.kt(0.0, 1.0, 0.55, 4)), abs=1e-16)
    assert c.kq[0] == pytest.approx(float(wb.kq(0.0, 1.0, 0.55, 4)), abs=1e-16)


def test_coefficients_equal_grouped_terms():
    """Coefficient k is the sum of the terms with J**k, evaluated exactly."""
    PD, EAR, Z = 0.93, 0.62, 5.0
    c = wb.cubic_coefficients(PD, EAR, Z)
    for terms, got in ((KT_TERMS, c.kt), (KQ_TERMS, c.kq)):
        for k in range(4):
            sub = [term for term in terms if term[1] == k]
            exact, scale = termwise_exact([(cc, 0, t, u, v) for cc, _, t, u, v in sub], 1.0, PD, EAR, Z)
            assert_machine_precision(got[k], exact, scale)


def test_cubic_reproduces_polynomials():
    rng = np.random.default_rng(4)
    PD = rng.uniform(0.5, 1.4, 40)
    EAR = rng.uniform(0.40, 1.00, 40)
    Z = np.full(40, 4.0)
    c = wb.cubic_coefficients(PD, EAR, Z)
    assert c.kt.shape == (40, 4)
    J = np.linspace(0, 1.4, 25)[:, None]
    np.testing.assert_allclose(wb.polyval_j(c.kt, J), wb.kt(J, PD, EAR, Z), rtol=0, atol=2e-15)
    np.testing.assert_allclose(wb.polyval_j(c.kq, J), wb.kq(J, PD, EAR, Z), rtol=0, atol=1e-15)


def test_propeller_object_matches_functions():
    prop = wb.BSeriesPropeller(PD=1.1, EAR=0.70, Z=4)
    J = np.linspace(0, 1.2, 50)
    np.testing.assert_allclose(prop.kt(J), wb.kt(J, 1.1, 0.70, 4), rtol=0, atol=2e-15)
    np.testing.assert_allclose(prop.kq(J), wb.kq(J, 1.1, 0.70, 4), rtol=0, atol=1e-15)
    np.testing.assert_allclose(prop.eta0(J[1:]), wb.eta0(J[1:], 1.1, 0.70, 4), rtol=1e-13)
    ow = prop.open_water(0.8)
    assert ow.kt == pytest.approx(float(wb.kt(0.8, 1.1, 0.70, 4)), abs=1e-15)
    assert "Z=4" in repr(prop)


def test_zero_thrust_from_numpy_roots():
    """np.roots takes decreasing powers: reverse the coefficients."""
    prop = wb.BSeriesPropeller(PD=1.0, EAR=0.70, Z=4)
    roots = np.roots(prop.kt_coefs[::-1])
    real = roots[np.isreal(roots)].real
    j0 = real[(real > 0) & (real < 2)].min()
    assert abs(float(prop.kt(j0))) < 1e-12
    assert 0.9 < j0 < 1.2  # zero thrust a little beyond J = P/D


def test_propeller_scalar_and_array_calls_agree():
    prop = wb.BSeriesPropeller(0.8, 0.85, 5)
    J = np.linspace(0, 1.0, 11)
    np.testing.assert_allclose([prop.kt(j) for j in J.tolist()], prop.kt(J), rtol=0, atol=1e-15)
    np.testing.assert_allclose([prop.kq(j) for j in J.tolist()], prop.kq(J), rtol=0, atol=1e-15)
    assert np.ndim(prop.kt(0.3)) == 0
