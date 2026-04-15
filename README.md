# potts-market-panic

Simulaciones de Monte Carlo del **modelo de Potts de *q* estados** aplicado a la modelización de mercados financieros, con un mecanismo de **campo de pánico global** para el colapso de burbujas especulativas y la dinámica de crisis sistémicas.

> Repositorio de código del Trabajo de Fin de Grado:  
> **"Modelo de Potts. Fundamentos y aplicaciones"**  
> Luis Fernando García Lobatón — Doble Grado en Física y Matemáticas  
> Universidad Complutense de Madrid, curso 2025–26  
> Tutor: Juan Pedro García Villaluenga

---

## Descripción

Este repositorio contiene el código completo utilizado para producir los resultados numéricos y las figuras del TFG. El modelo representa un mercado financiero como un sistema de Potts ferromagnético sobre una red cuadrada 2D con condiciones de contorno periódicas (topología de toro):

- Cada espín `σᵢ ∈ {1, …, q}` codifica la preferencia de inversión del agente `i` entre `q` activos.
- El acoplamiento ferromagnético `J > 0` captura el comportamiento gregario (*herding*).
- La temperatura `T` cuantifica el ruido de decisión de los agentes.
- Un **campo de pánico global** `η(t) ∈ [0, 1)` reduce el acoplamiento efectivo según `J_ef = J(1 − η)`, desplazando la temperatura crítica a `T_c^ef = T_c(1 − η)`.
- Una burbuja especulativa colapsa cuando `η` supera el umbral crítico `η_c = 1 − T/T_c`.

La dinámica se implementa con el algoritmo de Metropolis. En cada paso de Monte Carlo (PMC), se realizan `N = L²` intentos de actualización; para cada uno se propone un nuevo estado elegido **uniformemente al azar** entre los `q − 1` estados distintos del actual, garantizando la ergodicidad y el balance detallado.

---

## Requisitos

```
python >= 3.9
numpy
scipy
matplotlib
```

Instalación:

```bash
pip install numpy scipy matplotlib
```

---

## Uso

Generar todas las figuras:

```bash
python potts_simulaciones.py
```

Generar una sola figura (más rápido para depuración):

```bash
python potts_simulaciones.py 3    # solo la Figura 3 (instantáneas)
python potts_simulaciones.py 7    # solo la Figura 7 (hechos estilizados)
```

Todas las figuras se guardan en el directorio actual en formato `.pdf` y `.png`.

---

## Figuras generadas

| Archivo de salida | Contenido |
|---|---|
| `fig1_diagrama_fases.pdf` | Popularidad dominante `φ_{k*}(T)` para cuatro intensidades de pánico `η` |
| `fig2_colapso_eta.pdf` | Colapso de `φ_{k*}(η)` a temperatura fija, verificación de `η_c` |
| `fig3_snapshots.pdf` | Instantáneas de la red: transición de fase en equilibrio y colapso por shock |
| `fig4_shock_lehman.pdf` | Series temporales del índice con shock de pánico tipo Lehman Brothers |
| `fig5_lifetimes.pdf` | Derivación del umbral `φ_c`, distribución y ajuste exponencial de la duración de burbujas vs `η` |
| `fig6_retornos_colas.pdf` | Distribución de retornos y cola de ley de potencia (CCDF log-log) para tres escenarios |
| `fig7_stylized_facts.pdf` | Panel 2×2: verificación de los cuatro hechos estilizados fundamentales |

---

## Parámetros del modelo

Todos los parámetros se definen al inicio de cada bloque de figura y pueden modificarse libremente. Valores utilizados en el TFG:

| Parámetro | Símbolo | Valor |
|---|---|---|
| Número de estados (activos) | `q` | 5 |
| Tamaños de red | `L` | 12 – 18 |
| Constante de acoplamiento | `J` | 1.0 |
| Constante de Boltzmann | `k_B` | 1.0 (unidades reducidas) |
| Rango del campo de pánico | `η` | 0.00 – 0.80 |
| Pasos de termalización | — | 150 – 500 PMC |
| Pasos de medición | — | 200 – 5000 PMC |
| Realizaciones (lifetimes) | — | 15 – 30 |

La temperatura crítica exacta para `q = 5` en la red cuadrada 2D es:

```
T_c = J / (k_B · ln(1 + √5)) ≈ 0.8515 J/k_B
```

La elección `q = 5 > 4` garantiza una transición de fase de **primer orden** (coexistencia de fases, metaestabilidad), que es el ingrediente esencial para modelar la persistencia de las burbujas y la abruptez de los crashes.

---

## Resultados numéricos principales

### Diagrama de fases y campo de pánico

| Magnitud | Valor |
|---|---|
| `T_c(q = 5)` exacta | `0.8515 J/k_B` |
| `η_c(T = 0.90 T_c)` | `0.10` |
| `η_c(T = 0.88 T_c)` | `0.12` |
| Reducción del índice tras shock `η = 0.65` | **~93%** |

### Duración de burbujas (`T = 0.90 T_c`, `L = 12`, `φ_c = 0.308`)

El umbral de colapso `φ_c` se deriva estadísticamente de las distribuciones de `φ_{k*}` en cada fase: `φ_c = μ_dis + 2σ_dis`.

| `η` | `⟨τ⟩` (PMC) | SEM (PMC) |
|---|---|---|
| 0.00 | > 1500 | — |
| 0.10 | 1236 | 121 |
| 0.20 | 114 | 9 |
| 0.35 | 30 | 2 |
| 0.50 | 13 | 1 |
| 0.65 | 8 | 1 |
| 0.80 | 5 | < 1 |

Ajuste exponencial: `⟨τ⟩ = τ₀ · exp(−γ η)` con `γ ≈ 6.8`.

### Hechos estilizados (`T = 0.96 T_c`, `L = 16`, `η = 0`)

| Magnitud | Valor empírico | Modelo | Verificación |
|---|---|---|---|
| Exponente cola `μ` | `3.0 ± 0.1` | `3.17` | ✓ |
| Curtosis en exceso `κ` | 4 – 10 | `0.46` | Parcial (tamaño finito) |
| ACF retornos (`τ ≥ 2`) | `≲ 0.05` | `≲ 0.05` | ✓ |
| Agrupamiento volatilidad `γ_vol` | `0.10 – 0.40` | `0.18` | ✓ |
| Ratio asimetría gan.-pérd. | `1.5 – 2.0` | `1.08` | Parcial |

---

## Estructura del código

El archivo `potts_simulaciones.py` es autocontenido y está organizado del siguiente modo:

```
potts_simulaciones.py
│
├── Tc_ex(q)          Temperatura crítica exacta: T_c = J / (k_B ln(1+√q))
├── sweep()           Barrido de Metropolis: N = L² intentos, propuesta uniforme
├── mag()             Magnetización de Potts: m = (q·max_k N_k / N − 1) / (q − 1)
├── phi_max()         Popularidad dominante: φ_{k*} = max_k (N_k / N)
├── indice()          Índice de mercado: I(t) = φ_{k*} − 1/q
│
├── Figura 1  —  φ_{k*}(T/T_c) para η ∈ {0, 0.20, 0.40, 0.60}
├── Figura 2  —  φ_{k*}(η) para T/T_c ∈ {0.70, 0.85, 0.95}
├── Figura 3  —  Instantáneas de la red (L = 40): equilibrio + colapso por shock
├── Figura 4  —  Series temporales I(t) con shock η = 0.65
├── Figura 5  —  Umbral φ_c + duración de burbujas + ajuste exponencial
├── Figura 6  —  Distribución de retornos y CCDF para η ∈ {0, 0.25, 0.50}
└── Figura 7  —  Panel 2×2: colas pesadas, ACF retornos, ACF volatilidad, asimetría
```

---

## Fundamento físico

La dinámica se rige por el algoritmo de Metropolis con probabilidad de aceptación:

```
P_acc(σᵢ → σᵢ') = min(1, exp(−β ΔH))
```

donde `ΔH = −J_ef Σ_{j~i} [δ(σᵢ', σⱼ) − δ(σᵢ, σⱼ)]` y `J_ef = J(1 − η)`.

El campo de pánico desplaza la temperatura crítica de forma lineal:

```
T_c^ef(η) = T_c · (1 − η)
```

de modo que la intensidad de pánico necesaria para colapsar una burbuja que opera a temperatura `T` es exactamente:

```
η_c = 1 − T/T_c
```

Este resultado se deriva analíticamente sustituyendo `J → J_ef` en la expresión exacta de `T_c` obtenida por dualidad de Kramers-Wannier del modelo de Potts 2D (Wu, *Rev. Mod. Phys.* 54, 1982).

---

## Observables del mercado

| Observable | Definición | Interpretación |
|---|---|---|
| Popularidad `φ_k` | `N_k / N` | Fracción de agentes invertidos en el activo `k` |
| Popularidad dominante `φ_{k*}` | `max_k φ_k` | Concentración del mercado |
| Índice de mercado `I(t)` | `φ_{k*} − 1/q` | `≈ 0` en equilibrio, `> 0` en burbuja |
| Retorno `r(t)` | `I(t+1) − I(t)` | Serie temporal comparable con datos reales |

En la fase de burbuja, `φ_{k*} ≈ 1` (casi todos los agentes en un activo); en la fase desordenada, `φ_{k*} ≈ 1/q = 0.20`.

---

## Referencias

- **Sieczka, Sornette y Hołyst** (2011). The Lehman Brothers effect and bankruptcy cascades. *Eur. Phys. J. B* 82, 257–269.
- **Bornholdt** (2022). A q-spin Potts model of markets: Gain-loss asymmetry in stock indices as an emergent phenomenon. *Physica A* 588, 126565.
- **Wang et al.** (2018). Modeling and complexity of stochastic interacting Lévy type financial price dynamics. *Physica A* 499, 498–511.
- **Gopikrishnan et al.** (1999). Scaling of the distribution of fluctuations of financial market indices. *Phys. Rev. E* 60, 5305–5316.
- **Cont** (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quant. Finance* 1, 223–236.
- **Wu** (1982). The Potts model. *Rev. Mod. Phys.* 54, 235–268.
- **Metropolis et al.** (1953). Equation of state calculations by fast computing machines. *J. Chem. Phys.* 21, 1087–1092.

---

## Licencia

Licencia MIT. Véase el archivo `LICENSE` para más detalles.
