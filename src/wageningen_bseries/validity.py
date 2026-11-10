"""Extent of the Wageningen B-screw series and range checks.

The Oosterveld & van Oossanen (1975) polynomials are a regression on model tests.
Outside the parameter space that was tested they are extrapolations and can give
non-physical results, so this module lets the caller warn or fail on such inputs.

Sources for the extent
----------------------
* Blade-area ratios per blade number: S. Helma (2020), "Surprising behaviour of the
  Wageningen B-screw series polynomials", J. Mar. Sci. Eng. 8(3), 211, Table 1.
  https://doi.org/10.3390/jmse8030211 (open access, CC BY 4.0).
* Pitch ratio: the series covers 0.5 <= P/D <= 1.4; Helma (2020, Sec. 4.2) calls
  P/D = 1.4 the boundary of the available data.

Older summaries of the series list fewer models: no B2-38, no B5-90, and for six
blades only AE/A0 = 0.50, 0.65 and 0.80. See ``README.md`` for the details. No
source we found supports AE/A0 above 0.95 for six blades.
"""
from typing import Dict, List, Optional, Tuple
import math
import warnings

import numpy as np

__all__ = [
    "BSERIES_MODELS",
    "EAR_RANGE",
    "PD_RANGE",
    "Z_RANGE",
    "BSeriesRangeWarning",
    "BSeriesRangeError",
    "validity_issues",
    "check_validity",
]

#: Blade-area ratios AE/A0 of the series models for each blade number Z
#: (Helma 2020, Table 1).
BSERIES_MODELS: Dict[int, Tuple[float, ...]] = {
    2: (0.30, 0.38),
    3: (0.35, 0.50, 0.65, 0.80),
    4: (0.40, 0.55, 0.70, 0.85, 1.00),
    5: (0.45, 0.60, 0.75, 0.90, 1.05),
    6: (0.50, 0.65, 0.80, 0.95),
    7: (0.55, 0.70, 0.85),
}

#: Smallest and largest AE/A0 in the series for each blade number Z.
EAR_RANGE: Dict[int, Tuple[float, float]] = {
    z: (min(ears), max(ears)) for z, ears in BSERIES_MODELS.items()
}

#: Pitch-diameter ratio range of the series.
PD_RANGE: Tuple[float, float] = (0.5, 1.4)

#: Blade numbers in the series.
Z_RANGE: Tuple[int, int] = (2, 7)

_ACTIONS = ("ignore", "warn", "raise")

# Tolerance so that a value typed as 0.95 is not rejected because of float rounding.
_EPS = 1e-9


class BSeriesRangeWarning(UserWarning):
    """Inputs lie outside the tested range of the Wageningen B-series."""


class BSeriesRangeError(ValueError):
    """Inputs lie outside the tested range of the Wageningen B-series."""


def _describe(name: str, values: np.ndarray, mask: np.ndarray, bounds: str) -> str:
    bad = values[mask]
    n = bad.size
    first = float(bad.flat[0])
    extra = f" (and {n - 1} more)" if n > 1 else ""
    return f"{name}={first:g} outside {bounds}{extra}"


_SCALAR_TYPES = (int, float, np.integer, np.floating)


def _scalar_issues(PD: float, EAR: float, Z: float, J: Optional[float]) -> List[str]:
    """Same checks and messages as the array path, without NumPy overhead."""
    issues: List[str] = []
    zmin, zmax = Z_RANGE
    z_ok = math.isfinite(Z) and abs(Z - round(Z)) <= _EPS and zmin <= round(Z) <= zmax
    if not z_ok:
        issues.append(f"Z={Z:g} outside the integers {zmin}..{zmax}")
    lo, hi = PD_RANGE
    if not (lo - _EPS <= PD <= hi + _EPS):
        issues.append(f"P/D={PD:g} outside [{lo}, {hi}]")
    if z_ok:
        z = int(round(Z))
        elo, ehi = EAR_RANGE[z]
        if not (elo - _EPS <= EAR <= ehi + _EPS):
            issues.append(f"AE/A0={EAR:g} outside [{elo}, {ehi}] for Z={z}")
    if J is not None and not (J >= 0.0):
        issues.append(f"J={J:g} outside J >= 0")
    return issues


def validity_issues(PD, EAR, Z, J=None) -> List[str]:
    """Return human-readable descriptions of every out-of-range input.

    An empty list means all inputs lie inside the B-series. Arguments may be
    scalars or broadcastable arrays.
    """
    if all(isinstance(a, _SCALAR_TYPES) for a in (PD, EAR, Z)) and (
        J is None or isinstance(J, _SCALAR_TYPES)
    ):
        return _scalar_issues(
            float(PD), float(EAR), float(Z), None if J is None else float(J)
        )
    if J is None:
        PD, EAR, Z = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (PD, EAR, Z)))
    else:
        PD, EAR, Z, J = np.broadcast_arrays(
            *(np.asarray(a, dtype=float) for a in (PD, EAR, Z, J))
        )

    issues: List[str] = []

    zmin, zmax = Z_RANGE
    z_int = np.rint(Z)
    bad_z = (np.abs(Z - z_int) > _EPS) | (z_int < zmin) | (z_int > zmax) | ~np.isfinite(Z)
    if bad_z.any():
        issues.append(_describe("Z", Z, bad_z, f"the integers {zmin}..{zmax}"))

    lo, hi = PD_RANGE
    bad_pd = ~((PD >= lo - _EPS) & (PD <= hi + _EPS))
    if bad_pd.any():
        issues.append(_describe("P/D", PD, bad_pd, f"[{lo}, {hi}]"))

    # AE/A0 limits depend on Z; only meaningful where Z itself is valid.
    for z, (elo, ehi) in EAR_RANGE.items():
        sel = ~bad_z & (z_int == z)
        if not sel.any():
            continue
        bad_e = sel & ~((EAR >= elo - _EPS) & (EAR <= ehi + _EPS))
        if bad_e.any():
            issues.append(_describe("AE/A0", EAR, bad_e, f"[{elo}, {ehi}] for Z={z}"))

    if J is not None:
        bad_j = ~(J >= 0.0)
        if bad_j.any():
            issues.append(_describe("J", J, bad_j, "J >= 0"))

    return issues


def check_validity(
    PD, EAR, Z, J=None, *, action: Optional[str] = "warn", stacklevel: int = 2
) -> bool:
    """Check inputs against the extent of the B-series.

    Parameters
    ----------
    PD, EAR, Z : scalar or array_like
        Pitch-diameter ratio, expanded blade-area ratio AE/A0 and blade number.
    J : scalar or array_like, optional
        Advance coefficient; only ``J >= 0`` is checked.
    action : {"warn", "raise", "ignore"}
        What to do when something is out of range: emit a
        :class:`BSeriesRangeWarning`, raise a :class:`BSeriesRangeError`,
        or nothing. ``None`` is the same as ``"ignore"``.
    stacklevel : int
        Passed to :func:`warnings.warn`.

    Returns
    -------
    bool
        ``True`` when every input lies inside the series.
    """
    action = "ignore" if action is None else action
    if action not in _ACTIONS:
        raise ValueError(f"action must be one of {_ACTIONS}, got {action!r}")
    if action == "ignore":
        return not validity_issues(PD, EAR, Z, J)
    issues = validity_issues(PD, EAR, Z, J)
    if not issues:
        return True
    msg = "Outside the Wageningen B-series: " + "; ".join(issues)
    if action == "raise":
        raise BSeriesRangeError(msg)
    warnings.warn(msg, BSeriesRangeWarning, stacklevel=stacklevel)
    return False
