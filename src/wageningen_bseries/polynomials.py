"""Vectorised K_T, K_Q and open-water efficiency of the Wageningen B-screw series.

Every term of the Oosterveld & van Oossanen (1975) polynomials has an exponent of
J of at most 3. For a given propeller (P/D, AE/A0, Z) the thrust and torque
coefficients are therefore cubics in the advance coefficient J::

    K_T(J) = a0 + a1*J + a2*J**2 + a3*J**3
    K_Q(J) = b0 + b1*J + b2*J**2 + b3*J**3

This module computes the four coefficients of each cubic from a small matrix
product and evaluates the cubic with Horner's rule. The result is the same
polynomial as the term-by-term sum, up to floating-point rounding.
"""
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._coefficients import KQ_TERMS, KT_TERMS, Term
from .validity import check_validity

__all__ = [
    "kt",
    "kq",
    "eta0",
    "open_water",
    "cubic_coefficients",
    "polyval_j",
    "JCubic",
    "OpenWater",
    "BSeriesPropeller",
]


def _monomials(*tables: Sequence[Term]) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(sorted({(t, u, v) for table in tables for _, _, t, u, v in table}))


#: Distinct (t, u, v) exponents of (P/D, AE/A0, Z) used by K_T or K_Q.
_MONOS = _monomials(KT_TERMS, KQ_TERMS)
_MONO_INDEX = {m: i for i, m in enumerate(_MONOS)}


def _matrix(terms: Sequence[Term]) -> np.ndarray:
    """Coefficient matrix G with K(J) = sum_s J**s * (basis @ G.T)[..., s]."""
    g = np.zeros((4, len(_MONOS)))
    for c, s, t, u, v in terms:
        if s > 3:  # pragma: no cover - guarded by tests on the tables
            raise ValueError("J exponent above 3: the cubic reduction does not apply")
        g[s, _MONO_INDEX[(t, u, v)]] += c
    return g


_G_KT = _matrix(KT_TERMS)
_G_KQ = _matrix(KQ_TERMS)
_PD_POWERS = sorted({t for t, _, _ in _MONOS})
_EAR_POWERS = sorted({u for _, u, _ in _MONOS})
_Z_POWERS = sorted({v for _, _, v in _MONOS})


# Scalar path: the same matrices as rows of (coefficient, monomial index) pairs, in
# plain Python floats. For single operating points this avoids NumPy call overhead.
def _rows(g: np.ndarray) -> Tuple[Tuple[Tuple[float, int], ...], ...]:
    return tuple(tuple((float(c), k) for k, c in enumerate(row) if c != 0.0) for row in g)


_ROWS_KT = _rows(_G_KT)
_ROWS_KQ = _rows(_G_KQ)
_SCALAR_TYPES = (int, float, np.integer, np.floating)


def _all_scalar(*args) -> bool:
    return all(isinstance(a, _SCALAR_TYPES) for a in args)


def _basis_scalar(PD: float, EAR: float, Z: float) -> list:
    p = {e: PD**e for e in _PD_POWERS}
    a = {e: EAR**e for e in _EAR_POWERS}
    z = {e: Z**e for e in _Z_POWERS}
    return [p[t] * a[u] * z[v] for t, u, v in _MONOS]


def _cubic_scalar(rows, basis) -> Tuple[float, float, float, float]:
    c0, c1, c2, c3 = (sum([c * basis[k] for c, k in row]) for row in rows)
    return c0, c1, c2, c3


def _horner(c, J: float) -> float:
    return c[0] + J * (c[1] + J * (c[2] + J * c[3]))


def _powers(x: np.ndarray, exps: Sequence[int]) -> dict:
    out = {0: np.ones_like(x)}
    for e in exps:
        if e:
            out[e] = x**e
    return out


def _basis(PD, EAR, Z) -> np.ndarray:
    """Monomials (P/D)**t * (AE/A0)**u * Z**v stacked on the last axis."""
    PD, EAR, Z = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (PD, EAR, Z)))
    p = _powers(PD, _PD_POWERS)
    e = _powers(EAR, _EAR_POWERS)
    z = _powers(Z, _Z_POWERS)
    return np.stack([p[t] * e[u] * z[v] for t, u, v in _MONOS], axis=-1)


def _check(check, PD, EAR, Z, J=None) -> None:
    if check is not None and check != "ignore":
        # stacklevel 4: warn -> check_validity -> _check -> public function -> caller
        check_validity(PD, EAR, Z, J, action=check, stacklevel=4)


def _scalar(x: np.ndarray):
    """Return a NumPy scalar for 0-d results and the array otherwise."""
    return x[()] if x.ndim == 0 else x


class JCubic(NamedTuple):
    """Coefficients of K_T(J) and K_Q(J) in increasing powers of J.

    ``kt[..., k]`` multiplies ``J**k`` (k = 0..3); the same for ``kq``.
    """

    kt: np.ndarray
    kq: np.ndarray


class OpenWater(NamedTuple):
    """Open-water characteristics at one or more operating points."""

    kt: np.ndarray
    kq: np.ndarray
    eta0: np.ndarray


def cubic_coefficients(PD, EAR, Z, *, check: Optional[str] = "warn") -> JCubic:
    """Coefficients of the cubics K_T(J) and K_Q(J) for fixed P/D, AE/A0 and Z.

    Parameters
    ----------
    PD, EAR, Z : scalar or array_like
        Pitch-diameter ratio, expanded blade-area ratio AE/A0 and number of
        blades. They broadcast against each other.
    check : {"warn", "raise", "ignore"}
        Behaviour for inputs outside the B-series, see
        :func:`~wageningen_bseries.check_validity`.

    Returns
    -------
    JCubic
        ``(kt, kq)``, each an array of shape ``broadcast(PD, EAR, Z).shape + (4,)``
        with coefficients in increasing powers of J, so that
        ``K_T(J) = kt[..., 0] + kt[..., 1]*J + kt[..., 2]*J**2 + kt[..., 3]*J**3``.
        Use :func:`polyval_j` to evaluate them. Note that :func:`numpy.roots`
        expects decreasing powers: ``np.roots(kt[::-1])`` for a single propeller.
    """
    _check(check, PD, EAR, Z)
    b = _basis(PD, EAR, Z)
    return JCubic(kt=b @ _G_KT.T, kq=b @ _G_KQ.T)


def polyval_j(coefs, J):
    """Evaluate cubic coefficients from :func:`cubic_coefficients` at J.

    ``coefs`` has shape ``(..., 4)`` (increasing powers of J); ``J`` broadcasts
    against ``coefs[..., 0]``.
    """
    c = np.asarray(coefs, dtype=float)
    J = np.asarray(J, dtype=float)
    out = c[..., 0] + J * (c[..., 1] + J * (c[..., 2] + J * c[..., 3]))
    return _scalar(np.asarray(out))


def kt(J, PD, EAR, Z, *, check: Optional[str] = "warn"):
    """Thrust coefficient K_T = T / (rho n^2 D^4) of a B-series propeller.

    Parameters
    ----------
    J : scalar or array_like
        Advance coefficient J = V_A / (n D).
    PD : scalar or array_like
        Pitch-diameter ratio P/D.
    EAR : scalar or array_like
        Expanded blade-area ratio AE/A0.
    Z : scalar or array_like
        Number of blades.
    check : {"warn", "raise", "ignore"}
        Behaviour for inputs outside the B-series.

    All inputs broadcast against each other. Scalars give a NumPy scalar.
    The polynomials hold for a Reynolds number of 2e6; no scale correction
    is applied.
    """
    _check(check, PD, EAR, Z, J)
    if _all_scalar(J, PD, EAR, Z):
        b = _basis_scalar(float(PD), float(EAR), float(Z))
        return np.float64(_horner(_cubic_scalar(_ROWS_KT, b), float(J)))
    c = _basis(PD, EAR, Z) @ _G_KT.T
    return polyval_j(c, J)


def kq(J, PD, EAR, Z, *, check: Optional[str] = "warn"):
    """Torque coefficient K_Q = Q / (rho n^2 D^5) of a B-series propeller.

    Same arguments as :func:`kt`.
    """
    _check(check, PD, EAR, Z, J)
    if _all_scalar(J, PD, EAR, Z):
        b = _basis_scalar(float(PD), float(EAR), float(Z))
        return np.float64(_horner(_cubic_scalar(_ROWS_KQ, b), float(J)))
    c = _basis(PD, EAR, Z) @ _G_KQ.T
    return polyval_j(c, J)


def _eta0(J, KT, KQ):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _scalar(np.asarray(np.asarray(J, dtype=float) * KT / (2.0 * math.pi * KQ)))


def eta0(J, PD, EAR, Z, *, check: Optional[str] = "warn"):
    """Open-water efficiency eta_0 = J K_T / (2 pi K_Q).

    Same arguments as :func:`kt`. Where K_Q = 0 the result is ``inf`` or ``nan``.
    """
    return open_water(J, PD, EAR, Z, check=check).eta0


def open_water(J, PD, EAR, Z, *, check: Optional[str] = "warn") -> OpenWater:
    """K_T, K_Q and eta_0 together, sharing the work of computing the basis.

    Same arguments as :func:`kt`. Returns an :class:`OpenWater` named tuple.
    """
    _check(check, PD, EAR, Z, J)
    if _all_scalar(J, PD, EAR, Z):
        b = _basis_scalar(float(PD), float(EAR), float(Z))
        J = float(J)
        KT = np.float64(_horner(_cubic_scalar(_ROWS_KT, b), J))
        KQ = np.float64(_horner(_cubic_scalar(_ROWS_KQ, b), J))
    else:
        b = _basis(PD, EAR, Z)
        KT = polyval_j(b @ _G_KT.T, J)
        KQ = polyval_j(b @ _G_KQ.T, J)
    return OpenWater(KT, KQ, _eta0(J, KT, KQ))


class BSeriesPropeller:
    """One B-series propeller with its K_T and K_Q cubics in J precomputed.

    Use this when you evaluate the same propeller at many advance coefficients,
    for example inside a root finder: each call is a single Horner evaluation.

    >>> prop = BSeriesPropeller(PD=1.0, EAR=0.55, Z=4)
    >>> round(float(prop.kt(0.7)), 4)
    0.1807
    """

    __slots__ = ("PD", "EAR", "Z", "kt_coefs", "kq_coefs", "_kt", "_kq")

    def __init__(self, PD: float, EAR: float, Z: float, *, check: Optional[str] = "warn"):
        self.PD = float(PD)
        self.EAR = float(EAR)
        self.Z = float(Z)
        _check(check, self.PD, self.EAR, self.Z)
        coefs = cubic_coefficients(self.PD, self.EAR, self.Z, check="ignore")
        #: K_T cubic coefficients, increasing powers of J.
        self.kt_coefs = coefs.kt
        #: K_Q cubic coefficients, increasing powers of J.
        self.kq_coefs = coefs.kq
        self._kt = tuple(float(c) for c in coefs.kt)
        self._kq = tuple(float(c) for c in coefs.kq)

    def __repr__(self) -> str:
        return f"BSeriesPropeller(PD={self.PD:g}, EAR={self.EAR:g}, Z={self.Z:g})"

    def kt(self, J):
        """Thrust coefficient at advance coefficient(s) J."""
        if isinstance(J, _SCALAR_TYPES):
            return np.float64(_horner(self._kt, float(J)))
        return polyval_j(self.kt_coefs, J)

    def kq(self, J):
        """Torque coefficient at advance coefficient(s) J."""
        if isinstance(J, _SCALAR_TYPES):
            return np.float64(_horner(self._kq, float(J)))
        return polyval_j(self.kq_coefs, J)

    def eta0(self, J):
        """Open-water efficiency at advance coefficient(s) J."""
        return _eta0(J, self.kt(J), self.kq(J))

    def open_water(self, J) -> OpenWater:
        """K_T, K_Q and eta_0 at advance coefficient(s) J."""
        KT, KQ = self.kt(J), self.kq(J)
        return OpenWater(KT, KQ, _eta0(J, KT, KQ))
