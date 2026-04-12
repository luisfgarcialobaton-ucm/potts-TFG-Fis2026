"""
Modelo de Potts de q estados con Campo de Pánico Global
========================================================
Simulaciones de Monte Carlo para el TFG:
  "Modelo de Potts. Fundamentos y aplicaciones"

Autor  : Luis Fernando García Lobatón
Tutor  : Juan Pedro García Villaluenga
Centro : Doble Grado en Física y Matemáticas
         Universidad Complutense de Madrid, 2025-26

Descripción
-----------
Implementa el modelo de Potts de q estados sobre una red cuadrada 2D con
condiciones de contorno periódicas (toro), modelando un mercado financiero
con N agentes y q activos.

El campo de pánico global η ∈ [0,1) reduce el acoplamiento efectivo:
    J_ef = J·(1 − η)
desplazando la temperatura crítica:
    T_c^ef = T_c·(1 − η)
La burbuja colapsa cuando η supera el umbral crítico η_c = 1 − T/T_c.

Figuras generadas
-----------------
  fig1_panico_fases.pdf    — Diagrama de fases m(T) para varios valores de η
  fig2_colapso_eta.pdf     — Colapso de la magnetización m(η) a T fija
  fig3_shock_lehman.pdf    — Serie temporal con shock de pánico tipo Lehman
  fig4_lifetimes.pdf       — Duración de burbujas vs η con ajuste exponencial
  fig5_fat_tails.pdf       — Distribución de retornos y cola de ley de potencia
  fig6_stylized_facts.pdf  — Panel 2×2: cuatro hechos estilizados

Dependencias: numpy, scipy, matplotlib

Uso
---
    python potts_panic_simulations.py

Las figuras se guardan en el directorio actual como .pdf y .png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew, kstest
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")
np.random.seed(20240601)

# ── Estilo de las figuras ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "figure.dpi": 150, "font.family": "serif",
})


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO DE LA SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def barrido_metropolis(sigma: np.ndarray, L: int, beta: float,
                       Jeff: float, q: int) -> np.ndarray:
    """
    Realiza un paso de Monte Carlo completo (N = L² intentos de actualización).

    En cada intento se selecciona un agente al azar y se propone el estado
    siguiente en orden cíclico. El cambio se acepta con la probabilidad de
    Metropolis: P_acc = min(1, exp(−β·ΔH)).

    Parámetros
    ----------
    sigma : array (L, L) de enteros   — configuración de espines, valores en {1,…,q}
    L     : int                        — tamaño lineal de la red
    beta  : float                      — temperatura inversa β = 1/(k_B·T)
    Jeff  : float                      — acoplamiento efectivo J_ef = J·(1 − η)
    q     : int                        — número de estados de Potts (= número de activos)

    Devuelve
    --------
    sigma : array (L, L) de enteros   — configuración actualizada
    """
    N = L * L
    for _ in range(N):
        # Selección aleatoria del agente
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        # Estado propuesto (siguiente estado en orden cíclico)
        s_nuevo = (sigma[i, j] % q) + 1
        # Variación de energía debida al flip propuesto
        vecinos = [
            sigma[(i - 1) % L, j], sigma[(i + 1) % L, j],
            sigma[i, (j - 1) % L], sigma[i, (j + 1) % L],
        ]
        delta_H = -Jeff * sum(
            int(s_nuevo == s) - int(sigma[i, j] == s) for s in vecinos
        )
        # Criterio de aceptación de Metropolis
        if delta_H <= 0 or np.random.rand() < np.exp(-beta * delta_H):
            sigma[i, j] = s_nuevo
    return sigma


def magnetizacion(sigma: np.ndarray, q: int) -> float:
    """
    Calcula el parámetro de orden del modelo de Potts.

    Definición: m = (q·max_k N_k/N − 1) / (q − 1), con m ∈ [−1/(q−1), 1].
    En la fase ordenada (burbuja) m ≈ 1; en la desordenada m ≈ 0.
    """
    conteos = np.bincount(sigma.ravel(), minlength=q + 2)[1: q + 1]
    return (q * int(conteos.max()) / sigma.size - 1) / (q - 1)


def indice_mercado(sigma: np.ndarray, q: int) -> float:
    """
    Calcula el índice de mercado del modelo.

    Definición: I(t) = max_k φ_k − 1/q, con φ_k = popularidad del activo k.
    Es nulo en la fase desordenada y positivo en la ferromagnética (burbuja).
    Basado en la propuesta de Bornholdt (Physica A 588, 2022).
    """
    conteos = np.bincount(sigma.ravel(), minlength=q + 2)[1: q + 1]
    return int(conteos.max()) / sigma.size - 1.0 / q


def temperatura_critica_exacta(q: int) -> float:
    """
    Temperatura crítica exacta del modelo de Potts 2D en la red cuadrada.

    Resultado exacto derivado mediante la dualidad de Kramers-Wannier:
        T_c = J / (k_B · ln(1 + √q))
    en unidades reducidas J/k_B.

    Referencia: Wu, Rev. Mod. Phys. 54, 235 (1982).
    """
    return 1.0 / np.log(1.0 + np.sqrt(float(q)))


# Temperaturas críticas exactas para q = 2, …, 11
Tc = {q: temperatura_critica_exacta(q) for q in range(2, 12)}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Diagrama de fases con campo de pánico
# Muestra cómo η desplaza T_c^ef = T_c·(1 − η) hacia temperaturas menores
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 1: diagrama de fases con campo de pánico …")

q, L = 5, 14
rango_T = np.linspace(0.50 * Tc[q], 1.10 * Tc[q], 13)
valores_eta = [0.0, 0.25, 0.50, 0.75]
colores_eta = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

fig1, ax1 = plt.subplots(figsize=(7, 5))

for eta, color in zip(valores_eta, colores_eta):
    Jeff = 1.0 * (1.0 - eta)
    magnetizaciones = []
    for T in rango_T:
        beta = 1.0 / T
        s = np.random.randint(1, q + 1, size=(L, L))
        # Termalización
        for _ in range(280):
            s = barrido_metropolis(s, L, beta, Jeff, q)
        # Medición
        medidas = []
        for _ in range(250):
            s = barrido_metropolis(s, L, beta, Jeff, q)
            medidas.append(magnetizacion(s, q))
        magnetizaciones.append(float(np.mean(medidas)))

    # Línea punteada vertical en T_c^ef / T_c = 1 − η
    ax1.axvline(1.0 - eta, color=color, ls=":", lw=1.0, alpha=0.55)
    ax1.plot(
        rango_T / Tc[q], magnetizaciones,
        "o-", color=color, ms=5, lw=1.8,
        label=f"$\\eta={eta:.2f}$  ($T_c^{{\\rm ef}}/T_c={1 - eta:.2f}$)",
    )

ax1.set_xlabel("$T / T_c^{(0)}$", fontsize=13)
ax1.set_ylabel("Magnetización $m$", fontsize=13)
ax1.set_title(
    f"Efecto del campo de pánico sobre el diagrama de fases\n"
    f"($q={q}$, $L={L}$). Punteadas: $T_c^{{\\rm ef}}(\\eta)/T_c=1-\\eta$.",
    fontsize=10,
)
ax1.legend(loc="upper right")
ax1.set_xlim(0.45, 1.15)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(alpha=0.25)
fig1.tight_layout()
fig1.savefig("fig1_panico_fases.pdf", bbox_inches="tight")
fig1.savefig("fig1_panico_fases.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 1 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Colapso de la magnetización m(η) a temperatura fija
# Verifica numéricamente que η_c^teor = 1 − T/T_c
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 2: colapso m(η) a temperatura fija …")

q, L = 5, 16
razones_T = [0.80, 0.90, 0.95]
colores_T  = ["#2ca02c", "#1f77b4", "#d62728"]
barrido_eta = np.linspace(0.0, 0.95, 13)

fig2, ax2 = plt.subplots(figsize=(7, 5))

for razon_T, color in zip(razones_T, colores_T):
    T_sim = razon_T * Tc[q]
    beta  = 1.0 / T_sim
    magnetizaciones = []
    for eta in barrido_eta:
        Jeff = 1.0 * (1.0 - eta)
        s = np.random.randint(1, q + 1, size=(L, L))
        for _ in range(300):
            s = barrido_metropolis(s, L, beta, Jeff, q)
        medidas = []
        for _ in range(250):
            s = barrido_metropolis(s, L, beta, Jeff, q)
            medidas.append(magnetizacion(s, q))
        magnetizaciones.append(float(np.mean(medidas)))

    # Campo de pánico crítico teórico: η_c = 1 − T/T_c
    eta_c = 1.0 - razon_T
    ax2.axvline(eta_c, color=color, ls=":", lw=1.0, alpha=0.55)
    ax2.plot(
        barrido_eta, magnetizaciones,
        "o-", color=color, ms=5, lw=1.8,
        label=f"$T/T_c={razon_T:.2f}$  ($\\eta_c={eta_c:.2f}$)",
    )

ax2.set_xlabel("Intensidad del campo de pánico $\\eta$", fontsize=13)
ax2.set_ylabel("Magnetización $m$", fontsize=13)
ax2.set_title(
    f"Colapso inducido por pánico ($q={q}$, $L={L}$).\n"
    f"Punteadas: $\\eta_c^{{\\rm teor}}=1-T/T_c$.",
    fontsize=10,
)
ax2.legend()
ax2.set_xlim(-0.02, 1.0)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig("fig2_colapso_eta.pdf", bbox_inches="tight")
fig2.savefig("fig2_colapso_eta.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 2 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Series temporales con shock de pánico tipo Lehman Brothers
# Compara la dinámica sin pánico vs. con shock instantáneo en t = t_shock
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 3: series temporales con shock de pánico …")

q, L    = 5, 16
T_sim   = 0.90 * Tc[q]
beta    = 1.0 / T_sim
t_total = 2000
t_shock = 1000          # instante del shock (análogo a la quiebra de Lehman)

fig3, ejes3 = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

for idx, (eta_shock, titulo, color) in enumerate([
    (0.0,  "Sin campo de pánico ($\\eta=0$ todo el tiempo)",                   "#1f77b4"),
    (0.70, "Con shock de pánico ($\\eta=0.70$ para $t\\geq t_{\\rm shock}$)",  "#d62728"),
]):
    # Condición inicial: termalización en la fase de burbuja
    s = np.random.randint(1, q + 1, size=(L, L))
    for _ in range(400):
        s = barrido_metropolis(s, L, beta, 1.0, q)

    serie_indice = []
    for t in range(t_total):
        # Acoplamiento efectivo: normal antes del shock, reducido después
        Jeff = 1.0 if (t < t_shock or eta_shock == 0.0) else 1.0 * (1.0 - eta_shock)
        s = barrido_metropolis(s, L, beta, Jeff, q)
        serie_indice.append(indice_mercado(s, q))

    serie_indice = np.array(serie_indice)
    ax = ejes3[idx]
    ax.plot(np.arange(t_total), serie_indice, lw=0.7, color=color, alpha=0.9)
    if eta_shock > 0:
        ax.axvline(t_shock, color="k", ls="--", lw=1.5,
                   label=f"$t_{{\\rm shock}}={t_shock}$")
        ax.legend(fontsize=9.5)
    ax.set_ylabel("Índice $\\mathcal{I}(t)$", fontsize=12)
    ax.set_title(titulo, fontsize=10)
    ax.set_ylim(-0.02, 0.82)
    ax.grid(alpha=0.2)
    print(f"  η={eta_shock}: ⟨I⟩ antes={serie_indice[:t_shock].mean():.3f}  "
          f"⟨I⟩ después={serie_indice[t_shock:].mean():.3f}")

ejes3[1].set_xlabel("Tiempo (pasos MC)", fontsize=12)
fig3.suptitle(
    f"Impacto del shock de pánico sobre la dinámica del índice\n"
    f"($q={q}$, $L={L}$, $T=0.90\\,T_c$)",
    fontsize=11, y=1.01,
)
fig3.tight_layout()
fig3.savefig("fig3_shock_lehman.pdf", bbox_inches="tight")
fig3.savefig("fig3_shock_lehman.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 3 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Duración de burbujas vs campo de pánico
# Ajuste exponencial: ⟨τ⟩ = τ₀·exp(−γ·η), compatible con la ley de Arrhenius
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 4: duración de burbujas vs campo de pánico …")

q, L      = 5, 14
T_sim     = 0.92 * Tc[q]
beta      = 1.0 / T_sim
umbral_phi = 0.43           # umbral de colapso de la burbuja: φ_max < φ_c
valores_eta_b = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
n_realizaciones  = 18
duraciones = {e: [] for e in valores_eta_b}

for eta in valores_eta_b:
    Jeff = 1.0 * (1.0 - eta)
    for _ in range(n_realizaciones):
        # Condición inicial completamente ordenada (burbuja perfecta)
        s = np.ones((L, L), dtype=int)
        colapsada = False
        for t in range(1800):
            s = barrido_metropolis(s, L, beta, Jeff, q)
            # Comprobar colapso cada 5 pasos
            if t % 5 == 0:
                phi_max = indice_mercado(s, q) + 1.0 / q
                if phi_max < umbral_phi:
                    duraciones[eta].append(t)
                    colapsada = True
                    break
        if not colapsada:
            duraciones[eta].append(1800)   # burbuja aún viva al final
    print(f"  η={eta:.2f}: ⟨τ⟩={np.mean(duraciones[eta]):.1f}  "
          f"σ={np.std(duraciones[eta]):.1f}  PMC")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 5))

# Panel izquierdo: diagramas de caja
bp = ax4a.boxplot(
    [duraciones[e] for e in valores_eta_b],
    labels=[f"{e:.2f}" for e in valores_eta_b],
    patch_artist=True,
    medianprops=dict(color="black", lw=2),
)
mapa_colores = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(valores_eta_b)))
for caja, color in zip(bp["boxes"], mapa_colores):
    caja.set_facecolor(color)
    caja.set_alpha(0.75)
ax4a.set_xlabel("Campo de pánico $\\eta$", fontsize=12)
ax4a.set_ylabel("Duración (pasos MC)", fontsize=12)
ax4a.set_title("Distribución de la duración de burbujas\n(18 realizaciones por $\\eta$)", fontsize=10)
ax4a.grid(alpha=0.25, axis="y")

# Panel derecho: media ± desviación estándar con ajuste exponencial
medias = [np.mean(duraciones[e]) for e in valores_eta_b]
desv   = [np.std(duraciones[e])  for e in valores_eta_b]
ax4b.errorbar(valores_eta_b, medias, yerr=desv,
              fmt="o-", color="#d62728", capsize=6, lw=2, ms=7,
              label="Media $\\pm$ $\\sigma$")
try:
    def decaimiento_exp(eta, tau0, gamma):
        return tau0 * np.exp(-gamma * eta)
    params, _ = curve_fit(decaimiento_exp, valores_eta_b, medias,
                          p0=[1800, 5], maxfev=3000)
    eta_ajuste = np.linspace(0, 0.75, 200)
    ax4b.plot(eta_ajuste, decaimiento_exp(eta_ajuste, *params),
              "--", color="#1f77b4", lw=1.8,
              label=f"Ajuste $\\tau\\propto e^{{-\\gamma\\eta}}$, $\\gamma={params[1]:.2f}$")
    print(f"  Ajuste exponencial: γ = {params[1]:.3f}")
except Exception as error:
    print(f"  Advertencia en el ajuste: {error}")

ax4b.set_yscale("log")
ax4b.set_xlabel("Campo de pánico $\\eta$", fontsize=12)
ax4b.set_ylabel("Duración media (pasos MC, escala log)", fontsize=12)
ax4b.set_title(
    f"Duración media vs $\\eta$\n"
    f"($q={q}$, $L={L}$, $T=0.92\\,T_c$, $\\phi_c={umbral_phi}$)",
    fontsize=10,
)
ax4b.legend()
ax4b.grid(alpha=0.25, which="both")
fig4.tight_layout()
fig4.savefig("fig4_lifetimes.pdf", bbox_inches="tight")
fig4.savefig("fig4_lifetimes.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 4 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 5 — Distribución de retornos y cola de ley de potencia
# Compara tres escenarios: sin pánico, pánico moderado y pánico severo
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 5: distribución de retornos y colas …")

q, L  = 5, 16
T_sim = 0.97 * Tc[q]
beta  = 1.0 / T_sim
n_pasos = 3500

escenarios = [
    (0.0,  "Sin pánico ($\\eta=0$)",         "#1f77b4"),
    (0.30, "Pánico moderado ($\\eta=0.30$)", "#ff7f0e"),
    (0.55, "Pánico severo ($\\eta=0.55$)",   "#d62728"),
]
retornos_por_escenario = {}

for eta, etiqueta, color in escenarios:
    Jeff = 1.0 * (1.0 - eta)
    s = np.random.randint(1, q + 1, size=(L, L))
    for _ in range(400):
        s = barrido_metropolis(s, L, beta, Jeff, q)
    serie = []
    for _ in range(n_pasos):
        s = barrido_metropolis(s, L, beta, Jeff, q)
        serie.append(indice_mercado(s, q))
    retornos = np.diff(np.array(serie))
    retornos_por_escenario[eta] = retornos
    # Test de Kolmogorov-Smirnov frente a la gaussiana
    ret_std = (retornos - retornos.mean()) / retornos.std()
    _, p_ks = kstest(ret_std, "norm")
    print(f"  η={eta:.2f}: κ={kurtosis(retornos):.3f}  asim={skew(retornos):.3f}  "
          f"p-valor KS={p_ks:.4f}")

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 5))

# Panel izquierdo: histogramas normalizados
for eta, etiqueta, color in escenarios:
    ret = retornos_por_escenario[eta]
    bins = np.linspace(-4 * ret.std(), 4 * ret.std(), 55)
    ax5a.hist(ret, bins=bins, density=True, alpha=0.45, color=color,
              label=f"{etiqueta} ($\\kappa={kurtosis(ret):.2f}$)")

# Gaussiana de referencia (η = 0)
ret0 = retornos_por_escenario[0.0]
x_gauss = np.linspace(ret0.min(), ret0.max(), 300)
gauss = np.exp(-0.5 * (x_gauss / ret0.std()) ** 2) / (ret0.std() * np.sqrt(2 * np.pi))
ax5a.plot(x_gauss, gauss, "k--", lw=2, label="Gaussiana (ref.)")
ax5a.set_xlabel("Retorno $r_{\\mathcal{I}}$", fontsize=12)
ax5a.set_ylabel("Densidad de probabilidad", fontsize=12)
ax5a.set_title("Distribución de retornos por nivel de pánico", fontsize=10)
ax5a.legend(fontsize=8)
ax5a.set_xlim(-4 * ret0.std(), 4 * ret0.std())
ax5a.grid(alpha=0.2)

# Panel derecho: CCDF en escala log-log
for eta, etiqueta, color in escenarios:
    ret = retornos_por_escenario[eta]
    abs_r = np.sort(np.abs(ret))
    ccdf  = 1.0 - np.arange(1, len(abs_r) + 1) / float(len(abs_r))
    mascara = (abs_r > 1e-4) & (ccdf > 0.01)
    ax5b.loglog(abs_r[mascara], ccdf[mascara], "-", color=color, lw=1.8,
                alpha=0.85, label=etiqueta)

# Ajuste de ley de potencia a la cola de η = 0
abs_r0 = np.sort(np.abs(retornos_por_escenario[0.0]))
ccdf0  = 1.0 - np.arange(1, len(abs_r0) + 1) / float(len(abs_r0))
mascara_cola = (abs_r0 > abs_r0[int(0.75 * len(abs_r0))]) & (ccdf0 > 0.005)
exponente_mu = 2.5
if mascara_cola.sum() > 5:
    try:
        def ley_potencia(x, a, mu): return a * x ** (-mu)
        params0, _ = curve_fit(ley_potencia, abs_r0[mascara_cola],
                               ccdf0[mascara_cola], p0=[0.1, 2.5], maxfev=3000)
        exponente_mu = float(params0[1])
        x_ajuste = np.linspace(abs_r0[mascara_cola].min(),
                               abs_r0[mascara_cola].max(), 100)
        ax5b.loglog(x_ajuste, ley_potencia(x_ajuste, *params0),
                    "k--", lw=1.8,
                    label=f"Ley de potencia $\\mu={exponente_mu:.2f}$")
        print(f"  Exponente de la cola (η=0): μ = {exponente_mu:.3f}")
    except Exception as error:
        print(f"  Advertencia en el ajuste de ley de potencia: {error}")

ax5b.set_xlabel("$|r_{\\mathcal{I}}|$", fontsize=12)
ax5b.set_ylabel("$P(|r_{\\mathcal{I}}|>x)$", fontsize=12)
ax5b.set_title("Cola de la distribución (escala log-log)", fontsize=10)
ax5b.legend(fontsize=8)
ax5b.grid(alpha=0.25, which="both")
fig5.tight_layout()
fig5.savefig("fig5_fat_tails.pdf", bbox_inches="tight")
fig5.savefig("fig5_fat_tails.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 5 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 6 — Panel 2×2: verificación de los cuatro hechos estilizados
#   (a) Distribución de retornos vs gaussiana  (colas pesadas)
#   (b) ACF de retornos r_t                   (ausencia de autocorrelación lineal)
#   (c) ACF de volatilidad |r_t|              (agrupamiento de volatilidad)
#   (d) CCDF separada positivos/negativos     (asimetría ganancia-pérdida)
# ══════════════════════════════════════════════════════════════════════════════
print("Generando Figura 6: panel de hechos estilizados …")

q, L  = 5, 18
T_sim = 0.97 * Tc[q]
beta  = 1.0 / T_sim
n_sf  = 5000

# Serie temporal larga sin campo de pánico
s = np.random.randint(1, q + 1, size=(L, L))
for _ in range(400):
    s = barrido_metropolis(s, L, beta, 1.0, q)
serie_sf = []
for _ in range(n_sf):
    s = barrido_metropolis(s, L, beta, 1.0, q)
    serie_sf.append(indice_mercado(s, q))

retornos_sf   = np.diff(np.array(serie_sf))
volatilidad   = np.abs(retornos_sf)
print(f"  Serie: κ={kurtosis(retornos_sf):.3f}  asim={skew(retornos_sf):.4f}")

# ── Funciones de autocorrelación (ACF) ──────────────────────────────────────
retardo_max = 80
mu_r   = retornos_sf.mean()
var_r  = max(((retornos_sf - mu_r) ** 2).mean(), 1e-15)
mu_vol = volatilidad.mean()
var_vol = max(((volatilidad - mu_vol) ** 2).mean(), 1e-15)

acf_retornos   = np.array([
    ((retornos_sf[:len(retornos_sf)-k] - mu_r) *
     (retornos_sf[k:] - mu_r)).mean() / var_r
    for k in range(retardo_max)
])
acf_volatilidad = np.array([
    ((volatilidad[:len(volatilidad)-k] - mu_vol) *
     (volatilidad[k:] - mu_vol)).mean() / var_vol
    for k in range(retardo_max)
])
print(f"  ACF(r, τ=1)={acf_retornos[1]:.4f}  "
      f"ACF(|r|, τ=1)={acf_volatilidad[1]:.4f}  "
      f"ACF(|r|, τ=20)={acf_volatilidad[20]:.4f}")

# Separación para la asimetría ganancia-pérdida
ret_positivos = np.sort(retornos_sf[retornos_sf > 0])
ret_negativos = np.sort(np.abs(retornos_sf[retornos_sf < 0]))

# ── Construcción del panel ───────────────────────────────────────────────────
fig6, ejes6 = plt.subplots(2, 2, figsize=(12, 8))
fig6.suptitle(
    f"Verificación de hechos estilizados — Modelo de Potts\n"
    f"($q={q}$, $L={L}$, $T=0.97\\,T_c$, {n_sf} pasos MC)",
    fontsize=11, y=1.01,
)

# (a) Distribución de retornos vs gaussiana ──────────────────────────────────
ax_a = ejes6[0, 0]
sigma_sf = retornos_sf.std()
ax_a.hist(retornos_sf, bins=np.linspace(-4*sigma_sf, 4*sigma_sf, 60),
          density=True, color="#1f77b4", alpha=0.6,
          label=f"Retornos MC ($\\kappa={kurtosis(retornos_sf):.2f}$)")
x_g = np.linspace(-4*sigma_sf, 4*sigma_sf, 300)
ax_a.plot(x_g,
          np.exp(-0.5*(x_g/sigma_sf)**2) / (sigma_sf * np.sqrt(2*np.pi)),
          "r-", lw=2, label="Gaussiana")
ax_a.set_xlabel("Retorno $r$", fontsize=11)
ax_a.set_ylabel("Densidad", fontsize=11)
ax_a.set_title("(a) Distribución de retornos vs Gaussiana\n(colas pesadas)", fontsize=10)
ax_a.legend(fontsize=9)
ax_a.grid(alpha=0.2)

# (b) ACF de retornos (debe ser ≈ 0 para τ ≥ 1) ──────────────────────────────
ax_b = ejes6[0, 1]
intervalo_confianza = 1.96 / np.sqrt(len(retornos_sf))
ax_b.bar(np.arange(1, 50), acf_retornos[1:50],
         color="#1f77b4", alpha=0.7, width=0.8)
ax_b.axhline( intervalo_confianza, color="r", ls="--", lw=1.3, label="IC 95%")
ax_b.axhline(-intervalo_confianza, color="r", ls="--", lw=1.3)
ax_b.axhline(0, color="k", lw=0.8)
ax_b.set_xlabel("Retardo $\\tau$", fontsize=11)
ax_b.set_ylabel("ACF($r_t$)", fontsize=11)
ax_b.set_title("(b) Autocorrelación de retornos\n($\\approx0$ para $\\tau\\geq1$)", fontsize=10)
ax_b.legend(fontsize=9)
ax_b.set_ylim(-0.18, 0.28)
ax_b.grid(alpha=0.2)

# (c) ACF de volatilidad |r_t| (agrupamiento) ────────────────────────────────
ax_c = ejes6[1, 0]
retardos_c = np.arange(1, retardo_max)
ax_c.plot(retardos_c, acf_volatilidad[1:retardo_max],
          "o-", color="#ff7f0e", ms=3.5, lw=1.2, alpha=0.8,
          label="ACF($|r_t|$)")
ax_c.axhline(0,   color="k", lw=0.8)
ax_c.axhline( intervalo_confianza, color="r", ls="--", lw=1.3, label="IC 95%")
ax_c.axhline(-intervalo_confianza, color="r", ls="--", lw=1.3)

exponente_gamma = 0.22
mascara_positiva = acf_volatilidad[3:retardo_max] > 0.005
retardos_ajuste  = retardos_c[2:][mascara_positiva]
acf_ajuste       = acf_volatilidad[3:retardo_max][mascara_positiva]
if len(retardos_ajuste) > 4:
    try:
        def potencia_acf(tau, a, gamma): return a * tau ** (-gamma)
        params_gamma, _ = curve_fit(potencia_acf, retardos_ajuste, acf_ajuste,
                                    p0=[0.5, 0.2], maxfev=3000)
        exponente_gamma = float(params_gamma[1])
        tau_plot = np.linspace(max(retardos_ajuste.min(), 2),
                               retardos_ajuste.max(), 200)
        ax_c.plot(tau_plot, potencia_acf(tau_plot, *params_gamma),
                  "--", color="#2ca02c", lw=1.8,
                  label=f"Ajuste $\\tau^{{-\\gamma}}$, $\\gamma={exponente_gamma:.2f}$")
        print(f"  Exponente agrupamiento de volatilidad: γ = {exponente_gamma:.3f}")
    except Exception as error:
        print(f"  Advertencia en el ajuste de γ: {error}")

ax_c.set_xlabel("Retardo $\\tau$", fontsize=11)
ax_c.set_ylabel("ACF($|r_t|$)", fontsize=11)
ax_c.set_title("(c) Agrupamiento de volatilidad", fontsize=10)
ax_c.legend(fontsize=9)
ax_c.set_ylim(-0.1, max(0.55, acf_volatilidad[1] * 1.05))
ax_c.grid(alpha=0.2)

# (d) Asimetría ganancia-pérdida ─────────────────────────────────────────────
ax_d = ejes6[1, 1]
n_pos = len(ret_positivos); n_neg = len(ret_negativos)
ax_d.semilogy(ret_positivos,
              1.0 - np.arange(1, n_pos + 1) / float(n_pos),
              "-", color="#2ca02c", lw=1.8,
              label="Retornos positivos $r>0$")
ax_d.semilogy(ret_negativos,
              1.0 - np.arange(1, n_neg + 1) / float(n_neg),
              "-", color="#d62728", lw=1.8,
              label="Caídas $|r|$ ($r<0$)")
ax_d.set_xlabel("$|r|$", fontsize=11)
ax_d.set_ylabel("$P(|r|>x)$ (escala log)", fontsize=11)
ax_d.set_title("(d) Asimetría ganancia-pérdida\n(las caídas superan a las subidas)", fontsize=10)
ax_d.legend(fontsize=9)
ax_d.grid(alpha=0.2, which="both")

fig6.tight_layout()
fig6.savefig("fig6_stylized_facts.pdf", bbox_inches="tight")
fig6.savefig("fig6_stylized_facts.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Figura 6 guardada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN DE RESULTADOS NUMÉRICOS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("RESUMEN DE RESULTADOS NUMÉRICOS")
print("=" * 60)
print(f"\nTemperatura crítica exacta T_c(q=5) = {Tc[5]:.5f} J/k_B")
print(f"Campo de pánico crítico η_c(T=0.92·T_c) = {1 - 0.92:.2f}")
print(f"Campo de pánico crítico η_c(T=0.90·T_c) = {1 - 0.90:.2f}")
print(f"Campo de pánico crítico η_c(T=0.80·T_c) = {1 - 0.80:.2f}")

print("\nDuración media de burbujas (q=5, L=14, T=0.92·T_c):")
for eta in valores_eta_b:
    lt = duraciones[eta]
    print(f"  η={eta:.2f}: ⟨τ⟩={np.mean(lt):.1f}  σ={np.std(lt):.1f}  PMC")

print(f"\nHechos estilizados (q={q}, L={L}, T=0.97·T_c, {n_sf} pasos):")
print(f"  Curtosis en exceso κ         = {kurtosis(retornos_sf):.3f}")
print(f"  Asimetría (skewness)         = {skew(retornos_sf):.4f}")
print(f"  ACF(r,   τ=1)               = {acf_retornos[1]:.4f}")
print(f"  ACF(r,   τ=10)              = {acf_retornos[10]:.4f}")
print(f"  ACF(|r|, τ=1)               = {acf_volatilidad[1]:.4f}")
print(f"  ACF(|r|, τ=10)              = {acf_volatilidad[10]:.4f}")
print(f"  ACF(|r|, τ=40)              = {acf_volatilidad[40]:.4f}")
print(f"  Exponente agrupamiento γ     = {exponente_gamma:.3f}")
print(f"  Exponente cola retornos μ    = {exponente_mu:.3f}")
ratio_asim = (float(np.mean(retornos_sf < -0.015)) /
              max(float(np.mean(retornos_sf >  0.015)), 1e-9))
print(f"  Ratio asimetría ganancias-pérdidas = {ratio_asim:.3f}")
print("\nTodas las figuras han sido guardadas correctamente.")
