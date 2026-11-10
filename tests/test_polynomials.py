"""K_T, K_Q and eta_0 against term-by-term references, to machine precision."""
import math

import numpy as np
import pytest

import wageningen_bseries as wb
from conftest import KQ_TERMS, KT_TERMS, assert_machine_precision, termwise_exact


def _random_inputs(n, seed, extrapolate=False):
    rng = np.random.default_rng(seed)
    if extrapolate:
        return (
            rng.uniform(-0.5, 2.5, n),
            rng.uniform(0.2, 2.0, n),
            rng.uniform(0.1, 1.5, n),
            rng.uniform(1.0, 9.0, n),
        )
    Z = rng.integers(2, 8, n).astype(float)
    lo = np.array([wb.EAR_RANGE[int(z)][0] for z in Z])
    hi = np.array([wb.EAR_RANGE[int(z)][1] for z in Z])
    return rng.uniform(0, 1.5, n), rng.uniform(0.5, 1.4, n), rng.uniform(lo, hi), Z


@pytest.mark.parametrize("extrapolate", [False, True])
def test_against_exact_rational_sum(extrapolate):
    J, PD, EAR, Z = _random_inputs(300, seed=1, extrapolate=extrapolate)
    KT = wb.kt(J, PD, EAR, Z, check="ignore")
    KQ = wb.kq(J, PD, EAR, Z, check="ignore")
    for name, terms, got in (("KT", KT_TERMS, KT), ("KQ", KQ_TERMS, KQ)):
        ref = [termwise_exact(terms, *args) for args in zip(J, PD, EAR, Z)]
        exact, scale = np.array(ref).T
        assert_machine_precision(got, exact, scale)


def test_against_termwise_reference_table(reference_rows):
    """Frozen values from an independent pure-Python term-by-term implementation."""
    J, PD, EAR, Z, KT_ref, KQ_ref = reference_rows.T
    KT = wb.kt(J, PD, EAR, Z, check="ignore")
    KQ = wb.kq(J, PD, EAR, Z, check="ignore")
    scale_kt = [termwise_exact(KT_TERMS, *a)[1] for a in zip(J, PD, EAR, Z)]
    scale_kq = [termwise_exact(KQ_TERMS, *a)[1] for a in zip(J, PD, EAR, Z)]
    assert_machine_precision(KT, KT_ref, scale_kt)
    assert_machine_precision(KQ, KQ_ref, scale_kq)
    # In absolute terms the two implementations agree to a few 1e-16.
    np.testing.assert_allclose(KT, KT_ref, rtol=0, atol=2e-15)
    np.testing.assert_allclose(KQ, KQ_ref, rtol=0, atol=1e-15)


def test_scalar_in_scalar_out():
    v = wb.kt(0.5, 1.0, 0.55, 4)
    assert np.ndim(v) == 0
    assert isinstance(float(v), float)
    assert np.ndim(wb.kq(0.5, 1.0, 0.55, 4)) == 0
    assert np.ndim(wb.eta0(0.5, 1.0, 0.55, 4)) == 0


def test_broadcasting():
    J = np.linspace(0, 1, 7)[:, None]  # (7, 1)
    PD = np.array([0.6, 0.8, 1.0, 1.2])  # (4,)
    KT = wb.kt(J, PD, 0.70, 4)
    assert KT.shape == (7, 4)
    for i, j in enumerate(J[:, 0]):
        for k, pd in enumerate(PD):
            assert KT[i, k] == pytest.approx(float(wb.kt(j, pd, 0.70, 4)), abs=1e-15)


def test_accepts_lists():
    from_lists = wb.kt([0.2, 0.4], [1.0, 1.0], [0.55, 0.55], [4, 4])
    # Lists and arrays take the same code path: identical results.
    np.testing.assert_array_equal(
        from_lists, wb.kt(np.array([0.2, 0.4]), np.ones(2), np.full(2, 0.55), np.full(2, 4))
    )
    # Scalars take another path; summation order may differ by an ulp or two.
    np.testing.assert_allclose(
        from_lists,
        [wb.kt(0.2, 1.0, 0.55, 4), wb.kt(0.4, 1.0, 0.55, 4)],
        rtol=0,
        atol=2e-15,
    )


def test_eta0_definition():
    J, PD, EAR, Z = _random_inputs(200, seed=2)
    J = np.maximum(J, 0.05)
    KT = wb.kt(J, PD, EAR, Z)
    KQ = wb.kq(J, PD, EAR, Z)
    np.testing.assert_allclose(wb.eta0(J, PD, EAR, Z), J * KT / (2 * math.pi * KQ), rtol=1e-14)


def test_eta0_is_zero_at_bollard_pull():
    assert wb.eta0(0.0, 1.0, 0.55, 4) == 0.0


def test_open_water_matches_individual_functions():
    J, PD, EAR, Z = _random_inputs(50, seed=3)
    ow = wb.open_water(J, PD, EAR, Z)
    np.testing.assert_array_equal(ow.kt, wb.kt(J, PD, EAR, Z))
    np.testing.assert_array_equal(ow.kq, wb.kq(J, PD, EAR, Z))
    np.testing.assert_array_equal(ow.eta0, wb.eta0(J, PD, EAR, Z))


def test_scalar_path_machine_precision():
    """Plain Python floats take a separate code path; check it the same way."""
    J, PD, EAR, Z = _random_inputs(200, seed=5, extrapolate=True)
    for args in zip(J.tolist(), PD.tolist(), EAR.tolist(), Z.tolist()):
        for terms, fn in ((KT_TERMS, wb.kt), (KQ_TERMS, wb.kq)):
            exact, scale = termwise_exact(terms, *args)
            assert_machine_precision(fn(*args, check="ignore"), exact, scale)


def test_scalar_and_array_paths_agree():
    J, PD, EAR, Z = _random_inputs(100, seed=6)
    KT_arr = wb.kt(J, PD, EAR, Z)
    ow_arr = wb.open_water(J, PD, EAR, Z)
    for i, args in enumerate(zip(J.tolist(), PD.tolist(), EAR.tolist(), Z.astype(int).tolist())):
        assert wb.kt(*args) == pytest.approx(KT_arr[i], rel=0, abs=2e-15)
        ow = wb.open_water(*args)
        assert ow.kq == pytest.approx(ow_arr.kq[i], rel=0, abs=1e-15)
        assert ow.eta0 == pytest.approx(ow_arr.eta0[i], rel=1e-13)
    # NumPy scalars and 0-d arrays work too.
    assert wb.kt(np.float64(0.5), np.float32(1.0), 0.55, np.int64(4)) == pytest.approx(
        float(wb.kt(np.array(0.5), np.array(1.0), 0.55, 4)), abs=1e-7
    )


def test_eta0_with_zero_torque_gives_inf_without_warning():
    """K_Q = 0 must give inf/nan quietly (pytest turns warnings into errors here)."""
    from wageningen_bseries.polynomials import _eta0

    assert _eta0(0.5, np.float64(0.1), np.float64(0.0)) == np.inf
    assert np.isnan(_eta0(0.0, np.float64(0.0), np.float64(0.0)))
    out = _eta0(np.array([0.5, 0.5]), np.array([0.1, 0.1]), np.array([0.0, 0.02]))
    assert out[0] == np.inf and np.isfinite(out[1])
