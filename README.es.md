# wageningen-bseries

[![tests](https://github.com/lazgomez007/wageningen-bseries/actions/workflows/tests.yml/badge.svg)](https://github.com/lazgomez007/wageningen-bseries/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*English version: [README.md](README.md)*

Esta librería calcula el coeficiente de empuje K<sub>T</sub>, el coeficiente de par
K<sub>Q</sub> y la eficiencia en aguas libres η<sub>0</sub> de la **serie B de hélices de
Wageningen**. Usa los polinomios de regresión de Oosterveld y van Oossanen (1975):
39 términos para K<sub>T</sub> y 47 para K<sub>Q</sub>. Es pequeña, solo depende de NumPy
y está vectorizada.

```
K_T = Σ C_n · J^s · (P/D)^t · (AE/A0)^u · Z^v      (39 términos)
K_Q = Σ C_n · J^s · (P/D)^t · (AE/A0)^u · Z^v      (47 términos)
η_0 = J · K_T / (2π · K_Q)
```

Ningún término eleva J a una potencia mayor que 3. **Para una hélice dada (P/D,
AE/A0, Z), K<sub>T</sub>(J) y K<sub>Q</sub>(J) son cúbicas en J.** La librería calcula una
sola vez los cuatro coeficientes de cada cúbica y la evalúa con el esquema de Horner.
El polinomio es el mismo, salvo el redondeo de coma flotante. Esto es mucho más rápido
cuando la misma hélice se evalúa en muchos coeficientes de avance, por ejemplo dentro
de un buscador de raíces.

## Instalación

```bash
pip install git+https://github.com/lazgomez007/wageningen-bseries
```

También se puede instalar desde un clon con `pip install .`. Requiere Python ≥ 3.10
y NumPy ≥ 1.22.

## Uso

```python
import numpy as np
import wageningen_bseries as wb

# Hélice B4-55 (Z = 4, AE/A0 = 0.55), P/D = 1.0, con J = 0.7
wb.kt(0.7, PD=1.0, EAR=0.55, Z=4)    # 0.18073...
wb.kq(0.7, PD=1.0, EAR=0.55, Z=4)    # 0.03090...
wb.eta0(0.7, PD=1.0, EAR=0.55, Z=4)  # 0.65162...

# Las entradas se combinan como arrays de NumPy (broadcasting):
# un diagrama de aguas libres completo en una sola llamada
J = np.linspace(0, 1.2, 121)[:, None]
PD = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
ow = wb.open_water(J, PD, 0.70, 4)   # ow.kt, ow.kq y ow.eta0 tienen forma (121, 5)
```

### La cúbica en J

```python
c = wb.cubic_coefficients(PD=1.0, EAR=0.55, Z=4)
c.kt   # [a0, a1, a2, a3]  ->  K_T(J) = a0 + a1 J + a2 J² + a3 J³
c.kq   # [b0, b1, b2, b3]  ->  K_Q(J) = b0 + b1 J + b2 J² + b3 J³
wb.polyval_j(c.kt, 0.7)    # 0.18073...

# Lo mismo, preparado para evaluar muchas veces una misma hélice
prop = wb.BSeriesPropeller(PD=1.0, EAR=0.55, Z=4)
prop.kt(0.7), prop.kq(0.7), prop.eta0(0.7)
```

Los coeficientes van en potencias **crecientes** de J. `numpy.roots` espera potencias
decrecientes, así que hay que invertirlos: `np.roots(c.kt[::-1])`.

`cubic_coefficients` también admite arrays de P/D, AE/A0 o Z y devuelve forma `(..., 4)`.

### Ejemplo: empuje, diámetro y velocidad de avance conocidos

Si se conocen T, D y V<sub>A</sub> pero no las revoluciones, el cociente
K<sub>T</sub>/J² = T / (ρ V<sub>A</sub>² D²) es fijo. Entonces
K<sub>T</sub>(J) − (K<sub>T</sub>/J²)·J² = 0 es una cúbica, y J sale directamente de sus
raíces:

```python
prop = wb.BSeriesPropeller(PD=1.004, EAR=0.55, Z=4)
ratio = 0.371310                     # K_T / J² de este problema
coefs = prop.kt_coefs.copy()
coefs[2] -= ratio                    # K_T(J) - ratio * J² = 0
roots = np.roots(coefs[::-1])
J = roots[np.isreal(roots)].real
J = J[J > 0].min()                   # 0.7007
prop.open_water(J)                   # K_T 0.1823, K_Q 0.03124, eta0 0.6509
```

Esto reproduce la «solución exacta» que da Helma (2020, tabla A3) para el ejemplo 1 de
*The Wageningen Propeller Series* de Kuiper.

### Comprobación de rangos

Los polinomios están ajustados a ensayos con modelos, y fuera del rango ensayado
extrapolan. Todas las funciones aceptan `check="warn"` (el valor por defecto),
`"raise"` o `"ignore"`:

```python
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4)                  # BSeriesRangeWarning: P/D=1.6 outside [0.5, 1.4]
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4, check="raise")   # BSeriesRangeError (es un ValueError)
wb.kt(0.5, PD=1.6, EAR=0.55, Z=4, check="ignore")  # sin comprobación

wb.check_validity(PD=1.0, EAR=0.90, Z=6)           # True
wb.validity_issues(PD=1.6, EAR=1.05, Z=6, J=0.5)
# ['P/D=1.6 outside [0.5, 1.4]', 'AE/A0=1.05 outside [0.5, 0.95] for Z=6']
```

## Rango de validez

| Z | AE/A0 de los modelos de la serie | Rango de AE/A0 comprobado |
|---|---|---|
| 2 | 0.30, 0.38 | 0.30 – 0.38 |
| 3 | 0.35, 0.50, 0.65, 0.80 | 0.35 – 0.80 |
| 4 | 0.40, 0.55, 0.70, 0.85, 1.00 | 0.40 – 1.00 |
| 5 | 0.45, 0.60, 0.75, 0.90, 1.05 | 0.45 – 1.05 |
| 6 | 0.50, 0.65, 0.80, 0.95 | 0.50 – 0.95 |
| 7 | 0.55, 0.70, 0.85 | 0.55 – 0.85 |

* **Relaciones de área** según Helma (2020), tabla 1. Están en `wb.BSERIES_MODELS`
  y `wb.EAR_RANGE`.
* **Paso** 0.5 ≤ P/D ≤ 1.4 (`wb.PD_RANGE`). Helma (2020, §4.2) llama a P/D = 1.4 el
  límite de los datos disponibles.
* **Z** debe ser un entero entre 2 y 7.
* **J ≥ 0.** Las curvas medidas en aguas libres terminan cerca del empuje nulo. Más
  allá, los polinomios dan K<sub>T</sub> < 0, que es una extrapolación.
* **Número de Reynolds 2·10<sup>6</sup>.** Los polinomios describen la serie escalada a
  ese número de Reynolds.

**Seis palas.** Los resúmenes más antiguos de la serie llegan solo hasta AE/A0 = 0.80
con Z = 6 (B6-50, B6-65 y B6-80), y además no incluyen B2-38 ni B5-90. Un ejemplo es la
tabla de los
[apuntes 2.611 de MIT OpenCourseWare (2006)](https://ocw.mit.edu/courses/2-611-marine-power-and-propulsion-fall-2006/a0b75e4570cbaebb07e9862996334217_02propeller_test.pdf).
Helma (2020) incluye además una B6-95. Esta librería usa la tabla de Helma. Si se
quiere quedar dentro de los modelos en que coinciden todas las fuentes, para seis palas
hay que mantener AE/A0 ≤ 0.80. No encontramos ninguna fuente que respalde AE/A0
mayor que 0.95 con Z = 6.

**Comportamiento conocido de los polinomios.** Helma (2020) muestra que, con P/D alto y
pocas palas, los polinomios de 1975 pueden dar líneas de eficiencia máxima que se
doblan sobre sí mismas. Eso significa dos extremos, o ningún óptimo dentro de la serie.
Un optimizador basado en estos polinomios debería comprobarlo.

### Qué no incluye

* **Correcciones por número de Reynolds ΔK<sub>T</sub> y ΔK<sub>Q</sub>.** Oosterveld y
  van Oossanen (1975) también dan correcciones para números de Reynolds mayores que
  2·10<sup>6</sup>. No están implementadas. Todos los resultados son para
  Rn = 2·10<sup>6</sup>.
* Correcciones por espesor de pala, criterios de cavitación (Keller, Burrill) ni otras
  series, como las hélices en tobera.

## Verificación

`pytest` ejecuta estas comprobaciones (ver `tests/`):

* **Aritmética exacta.** K<sub>T</sub> y K<sub>Q</sub> se comparan con el polinomio
  evaluado en aritmética racional (`fractions.Fraction`). El error queda por debajo de
  8·ε por la suma de |términos| (ε = 2.2·10<sup>−16</sup>), tanto con arrays como con
  escalares.
* **Implementación independiente.** `tests/data/reference_termwise.csv` contiene 407
  puntos calculados con otra implementación, en Python puro y término a término, de los
  mismos polinomios. Ambas coinciden con un margen de 2·10<sup>−15</sup>.
* **Valores publicados.** Proceden de Helma (2020), J. Mar. Sci. Eng. 8(3), 211, que
  usa los mismos polinomios de 1975:
  * Tabla A3, «solución exacta» para una B4-55 con K<sub>T</sub>/J² = 0.371310:
    J = 0.7007, K<sub>T</sub> = 0.1823, K<sub>Q</sub> = 0.03124, η<sub>0</sub> = 0.6509,
    con el óptimo en P/D = 1.004. Se comprueba con los dígitos impresos.
  * Tabla A3, las tres iteraciones del programa de Kuiper, con ±1 en el último dígito
    impreso.
  * Sección 3.1, puntos de funcionamiento leídos en los diagramas de eficiencia de la
    B5-90 y la B3-80, con la precisión de una lectura gráfica.

## Rendimiento

Los tiempos son orientativos. Se midieron en un portátil con Python 3.12 y NumPy 2.5.

| Tarea | Término a término, Python puro | Esta librería |
|---|---|---|
| 100 000 puntos distintos (arrays) | 640 ms | 31 ms |
| Una hélice, un J (`BSeriesPropeller.kt`) | 6 µs | 0.2 µs |
| Una hélice, 100 000 valores de J | 640 ms | 0.5 ms |
| Un punto, `wb.kt(...)` escalar | 6 µs | 6.5 µs |

## Referencias

* M.W.C. Oosterveld y P. van Oossanen (1975). Further computer-analyzed data of the
  Wageningen B-screw series. *International Shipbuilding Progress* 22(251), 251–262.
  [doi:10.3233/ISP-1975-2225102](https://doi.org/10.3233/ISP-1975-2225102). Es la
  fuente de los coeficientes.
* S. Helma (2020). Surprising behaviour of the Wageningen B-screw series polynomials.
  *Journal of Marine Science and Engineering* 8(3), 211.
  [doi:10.3390/jmse8030211](https://doi.org/10.3390/jmse8030211). De acceso abierto.
  Es la fuente de la tabla de la serie y de los valores de los tests.
* G. Kuiper (1992). *The Wageningen Propeller Series*. MARIN, Wageningen.
* M.M. Bernitsas, D. Ray y P. Kinley (1981). *K<sub>T</sub>, K<sub>Q</sub> and efficiency
  curves for the Wageningen B-series propellers*. Informe 237, Department of Naval
  Architecture and Marine Engineering, University of Michigan.
  [hdl:2027.42/91702](https://hdl.handle.net/2027.42/91702). Contiene gráficas de los
  mismos polinomios.

Si usas esta librería en un trabajo publicado, cita a Oosterveld y van Oossanen (1975)
como fuente de los polinomios.

## Licencia

[MIT](LICENSE). Los coeficientes de regresión son datos publicados en la literatura
científica citada arriba.
