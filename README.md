# wageningen-bseries

[![tests](https://github.com/lazgomez007/wageningen-bseries/actions/workflows/tests.yml/badge.svg)](https://github.com/lazgomez007/wageningen-bseries/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Versión en español: [README.es.md](README.es.md)*

Thrust coefficient K<sub>T</sub>, torque coefficient K<sub>Q</sub> and open-water
efficiency η<sub>0</sub> of the **Wageningen B-screw propeller series**, from the
regression polynomials of Oosterveld & van Oossanen (1975): 39 terms for K<sub>T</sub>
and 47 for K<sub>Q</sub>. The library is small, depends only on NumPy, and is vectorised.

```
K_T = Σ C_n · J^s · (P/D)^t · (AE/A0)^u · Z^v      (39 terms)
K_Q = Σ C_n · J^s · (P/D)^t · (AE/A0)^u · Z^v      (47 terms)
η_0 = J · K_T / (2π · K_Q)
```

No term has J raised to a power above 3. **For a given propeller (P/D, AE/A0, Z),
K<sub>T</sub>(J) and K<sub>Q</sub>(J) are cubics in J.** The library computes the four
coefficients of each cubic once and evaluates it with Horner's rule. The result is the
same polynomial, to floating-point rounding, and it is much faster when the same
propeller is evaluated at many advance coefficients, as in a root finder.

## Install

```bash
pip install git+https://github.com/lazgomez007/wageningen-bseries
```

or, from a clone, `pip install .`. Requires Python ≥ 3.10 and NumPy ≥ 1.22.

## Usage

```python
import numpy as np
import wageningen_bseries as wb

# B4-55 propeller (Z = 4, AE/A0 = 0.55), P/D = 1.0, at J = 0.7
wb.kt(0.7, PD=1.0, EAR=0.55, Z=4)    # 0.18073...
wb.kq(0.7, PD=1.0, EAR=0.55, Z=4)    # 0.03090...
wb.eta0(0.7, PD=1.0, EAR=0.55, Z=4)  # 0.65162...

# Inputs broadcast like NumPy arrays: a whole open-water diagram in one call
J = np.linspace(0, 1.2, 121)[:, None]
PD = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
ow = wb.open_water(J, PD, 0.70, 4)   # ow.kt, ow.kq, ow.eta0 have shape (121, 5)
```

### The cubic in J

```python
c = wb.cubic_coefficients(PD=1.0, EAR=0.55, Z=4)
c.kt   # [a0, a1, a2, a3]  ->  K_T(J) = a0 + a1 J + a2 J² + a3 J³
c.kq   # [b0, b1, b2, b3]  ->  K_Q(J) = b0 + b1 J + b2 J² + b3 J³
wb.polyval_j(c.kt, 0.7)    # 0.18073...

# Same thing, packaged for repeated evaluation of one propeller
prop = wb.BSeriesPropeller(PD=1.0, EAR=0.55, Z=4)
prop.kt(0.7), prop.kq(0.7), prop.eta0(0.7)
```

Coefficients are in **increasing** powers of J. `numpy.roots` expects decreasing
powers, so reverse them: `np.roots(c.kt[::-1])`.

`cubic_coefficients` also broadcasts: with arrays of P/D, AE/A0 or Z it returns
shape `(..., 4)`.

### Example: known thrust, diameter and speed of advance

When T, D and V<sub>A</sub> are known but the rotation rate is not,
K<sub>T</sub>/J² = T / (ρ V<sub>A</sub>² D²) is fixed. Then K<sub>T</sub>(J) − (K<sub>T</sub>/J²)·J² = 0
is a cubic, and J follows directly from its roots:

```python
prop = wb.BSeriesPropeller(PD=1.004, EAR=0.55, Z=4)
ratio = 0.371310                     # K_T / J² for this design problem
coefs = prop.kt_coefs.copy()
coefs[2] -= ratio                    # K_T(J) - ratio * J² = 0
roots = np.roots(coefs[::-1])
J = roots[np.isreal(roots)].real
J = J[J > 0].min()                   # 0.7007
prop.open_water(J)                   # K_T 0.1823, K_Q 0.03124, eta0 0.6509
```

This reproduces the "exact solution" that Helma (2020, Table A3) gives for Example 1 of
Kuiper's *The Wageningen Propeller Series*.

### Validity checks

The polynomials are fitted to model tests. Outside the tested range they extrapolate.
Every function takes `check="warn"` (the default), `"raise"` or `"ignore"`:

```python
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4)                  # BSeriesRangeWarning: P/D=1.6 outside [0.5, 1.4]
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4, check="raise")   # BSeriesRangeError (a ValueError)
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4, check="ignore")  # no check

wb.check_validity(PD=1.0, EAR=0.90, Z=6)           # True
wb.validity_issues(PD=1.6, EAR=1.05, Z=6, J=0.5)
# ['P/D=1.6 outside [0.5, 1.4]', 'AE/A0=1.05 outside [0.5, 0.95] for Z=6']
```

## Range of validity

| Z | AE/A0 of the series models | AE/A0 range checked |
|---|---|---|
| 2 | 0.30, 0.38 | 0.30 – 0.38 |
| 3 | 0.35, 0.50, 0.65, 0.80 | 0.35 – 0.80 |
| 4 | 0.40, 0.55, 0.70, 0.85, 1.00 | 0.40 – 1.00 |
| 5 | 0.45, 0.60, 0.75, 0.90, 1.05 | 0.45 – 1.05 |
| 6 | 0.50, 0.65, 0.80, 0.95 | 0.50 – 0.95 |
| 7 | 0.55, 0.70, 0.85 | 0.55 – 0.85 |

* **Blade-area ratios** follow Helma (2020), Table 1. They are available as
  `wb.BSERIES_MODELS` and `wb.EAR_RANGE`.
* **Pitch ratio** 0.5 ≤ P/D ≤ 1.4 (`wb.PD_RANGE`). Helma (2020, §4.2) calls
  P/D = 1.4 the boundary of the available data.
* **Z** must be an integer from 2 to 7.
* **J ≥ 0.** The measured open-water curves end near zero thrust. Past that point
  the polynomials give K<sub>T</sub> < 0, which is an extrapolation.
* **Reynolds number 2·10<sup>6</sup>.** The polynomials describe the series scaled to
  this Reynolds number.

**Six blades.** Older summaries of the series stop at AE/A0 = 0.80 for Z = 6 (B6-50,
B6-65, B6-80), and they also leave out B2-38 and B5-90. One example is the table in the
[MIT OpenCourseWare 2.611 notes (2006)](https://ocw.mit.edu/courses/2-611-marine-power-and-propulsion-fall-2006/a0b75e4570cbaebb07e9862996334217_02propeller_test.pdf). Helma (2020) lists a B6-95 as well. This
library uses Helma's table. If you want to stay within the models that every source
agrees on, keep AE/A0 ≤ 0.80 for six blades. We found no source that supports
AE/A0 above 0.95 for Z = 6.

**Known behaviour of the polynomials.** Helma (2020) shows that at high P/D and low
blade numbers the 1975 polynomials can give lines of maximum efficiency that double
back. That means two extrema, or no optimum inside the series. An optimiser built on
these polynomials should check for this.

### Not included

* **Reynolds-number corrections ΔK<sub>T</sub>, ΔK<sub>Q</sub>.** Oosterveld & van
  Oossanen (1975) also give corrections for Reynolds numbers above 2·10<sup>6</sup>.
  They are not implemented. All results here are at Rn = 2·10<sup>6</sup>.
* Corrections for blade thickness, cavitation criteria (Keller, Burrill), and other
  series such as ducted propellers.

## Verification

`pytest` runs these checks (see `tests/`):

* **Exact arithmetic.** K<sub>T</sub> and K<sub>Q</sub> are compared with the polynomial
  evaluated in rational arithmetic (`fractions.Fraction`). The error stays below
  8·ε times the sum of |terms| (ε = 2.2·10<sup>−16</sup>), in both the array and the
  scalar code paths.
* **Independent implementation.** `tests/data/reference_termwise.csv` holds 407 points
  computed by a separate pure-Python term-by-term implementation of the same
  polynomials. The two agree to within 2·10<sup>−15</sup>.
* **Published values.** These come from Helma (2020), J. Mar. Sci. Eng. 8(3), 211,
  which uses the same 1975 polynomials:
  * Table A3, "exact solution" for a B4-55 propeller at K<sub>T</sub>/J² = 0.371310:
    J = 0.7007, K<sub>T</sub> = 0.1823, K<sub>Q</sub> = 0.03124, η<sub>0</sub> = 0.6509,
    with the optimum at P/D = 1.004. Checked to the printed digits.
  * Table A3, the three iterations of Kuiper's program, to ±1 in the last printed digit.
  * Section 3.1, operating points read from the efficiency diagrams of B5-90 and B3-80,
    to chart-reading accuracy.

## Performance

These timings are indicative. They were measured on one laptop with Python 3.12 and
NumPy 2.5.

| Task | Term-by-term, pure Python | This library |
|---|---|---|
| 100 000 different points (arrays) | 640 ms | 31 ms |
| One propeller, one J (`BSeriesPropeller.kt`) | 6 µs | 0.2 µs |
| One propeller, 100 000 J values | 640 ms | 0.5 ms |
| One point, scalar `wb.kt(...)` | 6 µs | 6.5 µs |

## References

* M.W.C. Oosterveld and P. van Oossanen (1975). Further computer-analyzed data of the
  Wageningen B-screw series. *International Shipbuilding Progress* 22(251), 251–262.
  [doi:10.3233/ISP-1975-2225102](https://doi.org/10.3233/ISP-1975-2225102). This is
  the source of the coefficients.
* S. Helma (2020). Surprising behaviour of the Wageningen B-screw series polynomials.
  *Journal of Marine Science and Engineering* 8(3), 211.
  [doi:10.3390/jmse8030211](https://doi.org/10.3390/jmse8030211). Open access. This is
  the source of the extent table and the test values.
* G. Kuiper (1992). *The Wageningen Propeller Series*. MARIN, Wageningen.
* M.M. Bernitsas, D. Ray and P. Kinley (1981). *K<sub>T</sub>, K<sub>Q</sub> and
  efficiency curves for the Wageningen B-series propellers*. Report 237, Department of
  Naval Architecture and Marine Engineering, University of Michigan.
  [hdl:2027.42/91702](https://hdl.handle.net/2027.42/91702). Plots of the same
  polynomials.

If you use this library in published work, please cite Oosterveld & van Oossanen (1975)
for the polynomials.

## License

[MIT](LICENSE). The regression coefficients are data published in the scientific
literature cited above.
