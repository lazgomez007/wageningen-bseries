"""Open-water polynomials of the Wageningen B-screw series.

K_T and K_Q from M.W.C. Oosterveld and P. van Oossanen (1975), "Further
computer-analyzed data of the Wageningen B-screw series", International
Shipbuilding Progress 22(251), 251-262, valid for Rn = 2e6.
"""
from ._coefficients import KQ_TERMS, KT_TERMS
from .polynomials import (
    BSeriesPropeller,
    JCubic,
    OpenWater,
    cubic_coefficients,
    eta0,
    kq,
    kt,
    open_water,
    polyval_j,
)
from .validity import (
    BSERIES_MODELS,
    EAR_RANGE,
    PD_RANGE,
    Z_RANGE,
    BSeriesRangeError,
    BSeriesRangeWarning,
    check_validity,
    validity_issues,
)

__version__ = "0.1.0"

__all__ = [
    "kt",
    "kq",
    "eta0",
    "open_water",
    "cubic_coefficients",
    "polyval_j",
    "BSeriesPropeller",
    "JCubic",
    "OpenWater",
    "KT_TERMS",
    "KQ_TERMS",
    "BSERIES_MODELS",
    "EAR_RANGE",
    "PD_RANGE",
    "Z_RANGE",
    "BSeriesRangeWarning",
    "BSeriesRangeError",
    "check_validity",
    "validity_issues",
    "__version__",
]
