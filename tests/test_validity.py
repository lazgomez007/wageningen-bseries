"""Range checks against the extent of the B-series."""
import warnings

import numpy as np
import pytest

import wageningen_bseries as wb
from wageningen_bseries import BSeriesRangeError, BSeriesRangeWarning


def test_extent_table():
    assert wb.EAR_RANGE == {
        2: (0.30, 0.38),
        3: (0.35, 0.80),
        4: (0.40, 1.00),
        5: (0.45, 1.05),
        6: (0.50, 0.95),
        7: (0.55, 0.85),
    }
    assert wb.PD_RANGE == (0.5, 1.4)
    assert wb.Z_RANGE == (2, 7)


@pytest.mark.parametrize("z", sorted(wb.BSERIES_MODELS))
def test_every_series_model_is_valid(z):
    for ear in wb.BSERIES_MODELS[z]:
        for pd in (0.5, 1.0, 1.4):
            assert wb.validity_issues(pd, ear, z, J=0.0) == []


def test_inside_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wb.kt(np.linspace(0, 1, 5), 1.0, 0.55, 4)
        wb.cubic_coefficients(0.8, 0.95, 6)


@pytest.mark.parametrize(
    "pd, ear, z, j, fragment",
    [
        (1.5, 0.55, 4, 0.5, "P/D=1.5"),
        (0.4, 0.55, 4, 0.5, "P/D=0.4"),
        (1.0, 1.05, 6, 0.5, "AE/A0=1.05 outside [0.5, 0.95] for Z=6"),
        (1.0, 0.30, 4, 0.5, "AE/A0=0.3"),
        (1.0, 0.55, 8, 0.5, "Z=8"),
        (1.0, 0.55, 4.5, 0.5, "Z=4.5"),
        (1.0, 0.55, 4, -0.1, "J=-0.1"),
    ],
)
def test_out_of_range_warns(pd, ear, z, j, fragment):
    with pytest.warns(BSeriesRangeWarning, match=fragment.replace(".", r"\.").replace("[", r"\[")):
        wb.kt(j, pd, ear, z)


def test_warning_points_at_caller():
    with pytest.warns(BSeriesRangeWarning) as rec:
        wb.kq(0.5, 1.6, 0.55, 4)
    assert rec[0].filename == __file__


def test_raise_mode():
    with pytest.raises(BSeriesRangeError, match="P/D"):
        wb.kt(0.5, 1.6, 0.55, 4, check="raise")
    with pytest.raises(BSeriesRangeError):
        wb.BSeriesPropeller(1.0, 1.05, 6, check="raise")
    with pytest.raises(ValueError):  # BSeriesRangeError is a ValueError
        wb.eta0(0.5, 1.0, 0.55, 9, check="raise")


def test_ignore_mode():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wb.kt(0.5, 1.6, 0.55, 4, check="ignore")
        wb.kt(0.5, 1.6, 0.55, 4, check=None)


def test_bad_action():
    with pytest.raises(ValueError, match="action"):
        wb.check_validity(1.0, 0.55, 4, action="loud")


def test_array_message_counts_offenders():
    PD = np.array([0.8, 1.6, 1.7, 1.0])
    issues = wb.validity_issues(PD, 0.55, 4)
    assert issues == ["P/D=1.6 outside [0.5, 1.4] (and 1 more)"]
    assert wb.check_validity(PD, 0.55, 4, action="ignore") is False
    assert wb.check_validity(1.0, 0.55, 4) is True


def test_ear_limit_depends_on_z():
    issues = wb.validity_issues(1.0, np.array([0.9, 0.9]), np.array([5, 6]))
    assert issues == []
    issues = wb.validity_issues(1.0, np.array([1.0, 1.0]), np.array([5, 6]))
    assert issues == ["AE/A0=1 outside [0.5, 0.95] for Z=6"]


@pytest.mark.parametrize(
    "pd, ear, z, j",
    [
        (1.6, 0.55, 4, 0.5),
        (1.0, 1.05, 6, 0.5),
        (1.0, 0.55, 4.5, -1.0),
        (0.3, 0.2, 2, 0.1),
        (1.0, 0.55, float("nan"), 0.5),
        (1.0, 0.55, 4, 0.5),
    ],
)
def test_scalar_and_array_checks_give_same_messages(pd, ear, z, j):
    scalar = wb.validity_issues(pd, ear, z, J=j)
    array = wb.validity_issues(np.array([pd]), np.array([ear]), np.array([z]), J=np.array([j]))
    assert scalar == array
