# potts-market-panic

Simulaciones de Monte Carlo del **modelo de Potts de $q$ estados** aplicado a la modelización de mercados financieros, con un mecanismo de campo de pánico global para el colapso de burbujas especulativas y la dinámica de crisis sistémicas.

> Repositorio de código del Trabajo de Fin de Grado:  
> **"Modelo de Potts. Fundamentos y aplicaciones"**  
> Luis Fernando García Lobatón — Doble Grado en Física y Matemáticas  
> Universidad Complutense de Madrid, curso 2025–26  
> Tutor: Juan Pedro García Villaluenga

---

## Descripción

Este repositorio contiene el código completo utilizado para producir los resultados numéricos y las figuras del TFG. El modelo representa un mercado financiero como un sistema de Potts ferromagnético sobre una red cuadrada 2D:

- Cada espín `σᵢ ∈ {1, …, q}` codifica la preferencia de inversión del agente `i` entre `q` activos.
- El acoplamiento ferromagnético `J > 0` captura el comportamiento gregario (*herding*).
- Un **campo de pánico global** `η(t) ∈ [0, 1)` reduce el acoplamiento efectivo `J_ef = J(1 − η)`, desplazando la temperatura crítica a `T_c^ef = T_c(1 − η)`.
- Una burbuja especulativa colapsa cuando `η` supera el umbral crítico `η_c = 1 − T/T_c`.

El script reproduce las seis figuras del trabajo y muestra por pantalla un resumen numérico de los resultados principales.

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

```bash
python potts_panic_simulations.py
```

Todas las figuras se guardan en el directorio actual en formato `.pdf` y `.png`.

| Archivo de salida | Contenido |
|---|---|
| `fig1_panico_fases.pdf` | Diagrama de fases `m(T)` para cuatro intensidades de pánico `η` |
| `fig2_colapso_eta.pdf` | Colapso de la magnetización `m(η)` a temperatura fija |
| `fig3_shock_lehman.pdf` | Series temporales del índice con shock de pánico tipo Lehman |
| `fig4_lifetimes.pdf` | Distribución de la duración de burbujas y ajuste exponencial vs `η` |
| `fig5_fat_tails.pdf` | Distribución de retornos y cola de ley de potencia (tres escenarios) |
| `fig6_stylized_facts.pdf` | Panel 2×2: cuatro hechos estilizados de los mercados financieros |

Al finalizar la ejecución, se imprime un resumen numérico con la temperatura crítica exacta, las estadísticas de duración de burbujas, el exponente de agrupamiento de volatilidad y el exponente de la cola de la distribución de retornos.

---

## Parámetros del modelo

Todos los parámetros se definen al inicio de cada bloque de figura y pueden modificarse libremente. Valores utilizados en el TFG:

| Parámetro | Símbolo | Valor |
|---|---|---|
| Número de estados de Potts (activos) | `q` | 5 |
| Tamaños de red | `L` | 14 – 22 |
| Constante de acoplamiento | `J` | 1.0 |
| Constante de Boltzmann | `k_B` | 1.0 (unidades reducidas) |
| Rango del campo de pánico | `η` | 0.0 – 0.75 |
| Pasos de termalización | `t_term` | 280 – 500 PMC |
| Pasos de medición | `t_med` | 250 – 5000 PMC |
| Realizaciones por punto | — | 13 – 18 |

La temperatura crítica exacta para `q = 5` en la red cuadrada 2D es `T_c = J / (k_B ln(1 + √5)) ≈ 0.8515 J/k_B`.

---

## Resultados numéricos principales

| Magnitud | Valor |
|---|---|
| `T_c(q=5)` exacta | `0.8515 J/k_B` |
| Campo de pánico crítico `η_c(T = 0.92 T_c)` | `0.08` |
| Duración media de burbuja con `η = 0.00` | `57.8 ± 25.7` PMC |
| Duración media de burbuja con `η = 0.30` | `11.1 ± 2.1` PMC |
| Duración media de burbuja con `η = 0.60` | `~5` PMC |
| Exponente del ajuste exponencial `γ` | `5.47` |
| Reducción del índice tras shock `η = 0.70` | `~80%` |
| Exponente de la cola de retornos `μ` | `3.12 ± 0.2` |
| Exponente de agrupamiento de volatilidad `γ` | `0.40` |

---

## Estructura del código

El archivo `potts_panic_simulations.py` es autocontenido y está organizado del siguiente modo:

```
potts_panic_simulations.py
│
├── metropolis_sweep()               Actualización MC: un barrido completo de N = L² intentos
├── magnetisation()                  Parámetro de orden de Potts m ∈ [-1/(q-1), 1]
├── market_index()                   Índice de mercado I(t) = max_k φ_k − 1/q
├── exact_critical_temperature()     T_c = J / (k_B ln(1+√q))
│
├── Figura 1  —  Diagrama de fases m(T/T_c) para η ∈ {0, 0.25, 0.50, 0.75}
├── Figura 2  —  Colapso m(η) para T/T_c ∈ {0.80, 0.90, 0.95}
├── Figura 3  —  Series temporales con shock de pánico instantáneo
├── Figura 4  —  Distribución de duración de burbujas + ajuste exponencial
├── Figura 5  —  Distribución de retornos y cola de ley de potencia (log-log)
└── Figura 6  —  Panel 2×2 de hechos estilizados
```

---

## Fundamento físico

La dinámica se rige por el algoritmo de Metropolis con probabilidad de aceptación

```
P_acc(σ → σ') = min(1, exp(−β ΔH))
```

donde `ΔH = −J_ef Σ_{j~i} [δ(σ', σ_j) − δ(σ, σ_j)]` y `J_ef = J(1 − η)`.

El campo de pánico desplaza la temperatura crítica de forma lineal: `T_c^ef(η) = T_c(1 − η)`, de modo que la intensidad de pánico necesaria para colapsar una burbuja que opera a temperatura `T` es exactamente `η_c = 1 − T/T_c`. Este resultado se deriva analíticamente mediante la dualidad de Kramers-Wannier del modelo de Potts 2D (Wu, Rev. Mod. Phys. 54, 1982).

---

## Referencias

- **Sieczka, Sornette y Hołyst** (2011). The Lehman Brothers effect and bankruptcy cascades. *Eur. Phys. J. B* 82, 257–269.
- **Bornholdt** (2022). A q-spin Potts model of markets. *Physica A* 588, 126565.
- **Wang et al.** (2018). Modeling and complexity of stochastic interacting Lévy type financial price dynamics. *Physica A* 499, 498–511.
- **Gopikrishnan et al.** (1999). Scaling of the distribution of fluctuations of financial market indices. *Phys. Rev. E* 60, 5305.
- **Cont** (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quant. Finance* 1, 223–236.
- **Wu** (1982). The Potts model. *Rev. Mod. Phys.* 54, 235–268.

---

## Licencia

Licencia MIT. Véase el archivo `LICENSE` para más detalles.
