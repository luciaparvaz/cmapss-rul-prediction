"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 1: Inspección y análisis exploratorio (sobre Fase 0)
=============================================================================
Esta versión sustituye a la Fase 1 original, que definía su propia carga
de datos (con rutas al sandbox original, inexistentes en este repo) y su
propia normalización — `normalize_per_unit`, que ajustaba un MinMaxScaler
POR MOTOR de forma independiente en train y test. Esa es la fuente del
sesgo de leakage que corrige Fase 0 (ver fase0_preprocesado.py y su
notebook para el diagnóstico cuantitativo). Aquí no se redefine nada de
eso: se importa de fase0_preprocesado.py, y esta fase se limita a lo que
le corresponde — inspección descriptiva y visualización exploratoria,
sin volver a tomar decisiones de preprocesado.

Contenido:
  - Estadísticas descriptivas del dataset (dimensiones, vida útil, nulos)
  - Resumen estadístico por sensor (media, std, min, max, CV%)
  - Visualización de la selección de sensores informativos (Fase 0)
  - Distribución de vida útil de los motores
  - Distribuciones de los sensores informativos
  - Efecto de la normalización GLOBAL sobre trayectorias reales de
    varios motores (reemplaza la demo de normalización per-motor de la
    versión original, que precisamente ilustraba el sesgo, no la
    corrección)
  - EXTENSIÓN (sept. 2026): inspección comparada de los 4 subdatasets
    (FD001-FD004) — dimensiones, nulos, sensores informativos y
    normalización por subdataset (global para FD001/FD003, por régimen
    k-means para FD002/FD004, ver fase0_preprocesado.py::normalize_dataset).
    El análisis profundo de FD001 de arriba no se toca; esto se añade
    como sección adicional al final.

Referencias:
  - Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
    propagation modeling for aircraft engine run-to-failure simulation.
    PHM 2008, Denver, CO.
  - Ramasso, E., & Saxena, A. (2014). Performance benchmarking and
    analysis of prognostic methods for CMAPSS datasets. IJPHM, 5(2).
    DOI: 10.36001/ijphm.2014.v5i2.2236.
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
np.random.seed(42)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fase0_preprocesado as f0

FIG_DIR    = f0.FIG_DIR
OUTPUT_DIR = f0.OUTPUT_DIR

PALETTE = {
    "primary": "#2563EB", "secondary": "#10B981", "accent": "#F59E0B",
    "danger": "#EF4444", "neutral": "#6B7280", "background": "#F8FAFC",
    "text": "#1E293B",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["background"], "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1", "axes.labelcolor": PALETTE["text"],
    "text.color": PALETTE["text"], "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"], "grid.color": "#E2E8F0",
    "grid.linestyle": "--", "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.dpi": 150,
})

SENSOR_NAMES = {
    "s1": "T2 — Temp. inlet fan", "s2": "T24 — Temp. LPC outlet",
    "s3": "T30 — Temp. HPC outlet", "s4": "T50 — Temp. LPT outlet",
    "s5": "P2 — Pressure fan inlet", "s6": "P15 — Pressure bypass",
    "s7": "P30 — Pressure HPC outlet", "s8": "Nf — Fan speed (physical)",
    "s9": "Nc — Core speed (physical)", "s10": "epr — Engine pressure ratio",
    "s11": "Ps30 — Static pressure HPC", "s12": "phi — Fuel flow ratio",
    "s13": "NRf — Corrected fan speed", "s14": "NRc — Corrected core speed",
    "s15": "BPR — Bypass ratio", "s16": "farB — Burner fuel-air ratio",
    "s17": "htBleed — Bleed enthalpy", "s18": "Nf_dmd — Demanded fan speed",
    "s19": "PCNfR_dmd — Demanded corrected fan speed",
    "s20": "W31 — HPT coolant bleed", "s21": "W32 — LPT coolant bleed",
}


# ─────────────────────────────────────────────
# 1. INSPECCIÓN DESCRIPTIVA
# ─────────────────────────────────────────────

def inspect_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     rul_df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas generales + resumen por sensor (train)."""
    print(f"\n{'DIMENSIONES':─<50}")
    print(f"  Train : {train_df.shape[0]:>8,} filas  ×  {train_df.shape[1]} columnas")
    print(f"  Test  : {test_df.shape[0]:>8,} filas  ×  {test_df.shape[1]} columnas")
    print(f"  RUL   : {rul_df.shape[0]:>8,} valores")

    n_train_eng = train_df["engine_id"].nunique()
    n_test_eng  = test_df["engine_id"].nunique()
    print(f"\n{'MOTORES':─<50}")
    print(f"  Train : {n_train_eng} motores  |  Test : {n_test_eng} motores")

    life = train_df.groupby("engine_id")["cycle"].max()
    print(f"\n{'VIDA ÚTIL POR MOTOR (Train)':─<50}")
    print(f"  Mínima/Máxima : {life.min()} / {life.max()} ciclos")
    print(f"  Media ± std   : {life.mean():.1f} ± {life.std():.1f} ciclos")
    print(f"  Mediana       : {life.median():.0f} ciclos")

    nulos_train = train_df.isnull().sum().sum()
    nulos_test  = test_df.isnull().sum().sum()
    print(f"\n{'VALORES NULOS':─<50}")
    print(f"  Train : {nulos_train}  |  Test : {nulos_test}")

    sensor_stats = train_df[f0.SENSOR_COLS].agg(["mean", "std", "min", "max"]).T
    sensor_stats.columns = ["Media", "Std", "Mín", "Máx"]
    sensor_stats["Varianza"] = sensor_stats["Std"] ** 2
    sensor_stats["CV_%"] = (sensor_stats["Std"] / sensor_stats["Media"].abs().replace(0, np.nan) * 100).round(2)
    sensor_stats.index.name = "Sensor"
    return sensor_stats


# ─────────────────────────────────────────────
# 2. VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_sensor_variance(train_df: pd.DataFrame, low_var: list, save_path: str = None):
    """Barras de std por sensor, marcando los descartados por Fase 0."""
    stds = train_df[f0.SENSOR_COLS].std().sort_values(ascending=True)
    colors = [PALETTE["danger"] if s in low_var else PALETTE["primary"] for s in stds.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(stds.index, stds.values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)
    ax.axvline(f0.STD_THRESHOLD, color=PALETTE["danger"], linestyle="--",
               linewidth=1.5, label=f"Umbral std = {f0.STD_THRESHOLD}")
    for bar, val in zip(bars, stds.values):
        ax.text(val + stds.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=PALETTE["text"])
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PALETTE["primary"], label="Informativo"),
        Patch(facecolor=PALETTE["danger"], label=f"No informativo (std < {f0.STD_THRESHOLD})"),
    ], loc="lower right", fontsize=9)
    ax.set_xlabel("Desviación típica (std)", fontsize=11)
    ax.set_title("Variabilidad de sensores — FD001 (Train)\n"
                 "Selección de sensores informativos calculada en Fase 0",
                 fontsize=12, pad=15)
    ax.grid(axis="x", alpha=0.5)
    ax.set_xlim(left=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_life_distribution(train_df: pd.DataFrame, save_path: str = None):
    life = train_df.groupby("engine_id")["cycle"].max()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(life.values, bins=20, color=PALETTE["primary"], alpha=0.75,
            edgecolor="white", linewidth=0.8, density=True, label="Histograma")
    try:
        kde = gaussian_kde(life.values)
        x_range = np.linspace(life.min() - 10, life.max() + 10, 300)
        ax.plot(x_range, kde(x_range), color=PALETTE["danger"], linewidth=2.5, label="KDE")
    except Exception:
        pass
    ax.axvline(life.mean(), color=PALETTE["accent"], linestyle="--", linewidth=2,
               label=f"Media = {life.mean():.0f} ciclos")
    ax.axvline(life.median(), color=PALETTE["secondary"], linestyle=":", linewidth=2,
               label=f"Mediana = {life.median():.0f} ciclos")
    ax.set_xlabel("Vida útil (ciclos hasta fallo)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución de vida útil de los motores — FD001 (Train)\n"
                 "100 motores | 1 condición operacional | Fallo: degradación HPC",
                 fontsize=12, pad=15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_sensor_distributions(train_df: pd.DataFrame, informative: list, save_path: str = None):
    n = len(informative)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes_flat = axes.flatten()
    for i, sensor in enumerate(informative):
        ax = axes_flat[i]
        ax.hist(train_df[sensor].dropna(), bins=40, color=PALETTE["primary"],
                alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{sensor}\n{SENSOR_NAMES[sensor]}", fontsize=8, pad=4)
        ax.set_xlabel("Valor", fontsize=7)
        ax.set_ylabel("Frecuencia", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Distribuciones de sensores informativos — FD001 (Train)",
                 fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_global_normalized_sample(train_df: pd.DataFrame, informative: list,
                                   global_scaler, n_motors: int = 5,
                                   save_path: str = None):
    """
    Muestra la señal normalizada GLOBALMENTE (Fase 0) de varios motores,
    para los mismos 3 sensores que usaba la Fase 1 original — pero con
    la corrección: como el escalador es el mismo para todos los
    motores, las diferencias de nivel entre motores en la gráfica son
    reales (reflejan degradación real), no un artefacto de que cada
    motor se reescale a su propio rango [0,1] como ocurría con
    `normalize_per_unit` (el bug corregido en Fase 0).
    """
    sensors_to_plot = [s for s in ["s2", "s3", "s4"] if s in informative] or informative[:3]
    train_norm = f0.apply_global_scaler(train_df, informative, global_scaler)
    sample_engines = train_norm["engine_id"].unique()[:n_motors]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["danger"], PALETTE["neutral"]]

    fig, axes = plt.subplots(1, len(sensors_to_plot), figsize=(14, 4), sharey=False)
    if len(sensors_to_plot) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors_to_plot):
        for j, eng in enumerate(sample_engines):
            sub = train_norm[train_norm["engine_id"] == eng]
            ax.plot(sub["cycle"], sub[sensor], color=colors[j], alpha=0.85,
                    linewidth=1.2, label=f"Motor {eng}")
        ax.set_title(f"{sensor} — {SENSOR_NAMES[sensor]}", fontsize=9)
        ax.set_xlabel("Ciclo de operación", fontsize=8)
        ax.set_ylabel("Valor (GlobalMinMaxScaler)", fontsize=8)
        ax.grid(alpha=0.35)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Señales normalizadas GLOBALMENTE (fit solo en train, Fase 0)\n"
                 "A diferencia de la normalización per-motor, el nivel entre motores es comparable",
                 fontsize=11, y=1.02, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 2b. INSPECCIÓN COMPARADA DE LOS 4 SUBDATASETS (extensión)
# ─────────────────────────────────────────────

def build_subsets_comparison(subsets: list = f0.SUBSETS) -> tuple:
    """
    Carga y describe los 4 subdatasets, aplicando a cada uno la
    normalización que le corresponde (f0.normalize_dataset: global para
    FD001/FD003, por régimen k-means para FD002/FD004 — nunca por
    motor). Recalcula sensores informativos por subdataset: no se
    reutiliza la lista de FD001, porque más condiciones operacionales
    implican más varianza real en sensores que en FD001 son ~constantes
    (ver resultados: FD002/FD004 retienen 20 sensores frente a los 14
    de FD001, ya que sensores como s1/s5/s6/s10/s18/s19 sí varían con
    la condición operacional aunque sean planos dentro de una única
    condición).

    Retorna
    -------
    summary_df : tabla comparativa (una fila por subdataset)
    loaded     : dict {subset: {"train", "test", "rul", "informative",
                                 "low_var", "norm_info"}}
    """
    rows = []
    loaded = {}
    for subset in subsets:
        meta = f0.SUBSET_META[subset]
        train_df, test_df, rul_df = f0.load_cmapss(subset)
        informative, low_var = f0.select_informative_sensors(train_df, subset=subset)
        _, _, norm_info = f0.normalize_dataset(train_df, test_df, informative, subset=subset)

        life = train_df.groupby("engine_id")["cycle"].max()
        nulos = int(train_df.isnull().sum().sum() + test_df.isnull().sum().sum())

        rows.append({
            "Subdataset": subset,
            "Motores train": train_df["engine_id"].nunique(),
            "Motores test": test_df["engine_id"].nunique(),
            "Ciclos train": len(train_df),
            "Ciclos test": len(test_df),
            "Cond. operacionales": meta["n_conditions"],
            "Modos de fallo": meta["n_fault_modes"],
            "Vida media (ciclos)": round(life.mean(), 1),
            "Vida std": round(life.std(), 1),
            "Nulos (train+test)": nulos,
            "Sensores informativos": len(informative),
            "Sensores descartados": len(low_var),
            "Normalización": norm_info["strategy"],
        })
        loaded[subset] = {
            "train": train_df, "test": test_df, "rul": rul_df,
            "informative": informative, "low_var": low_var,
            "norm_info": norm_info,
        }

    summary_df = pd.DataFrame(rows).set_index("Subdataset")
    return summary_df, loaded


def plot_subsets_comparison(summary_df: pd.DataFrame, loaded: dict, save_path: str = None):
    """
    Compara los 4 subdatasets en dos paneles:
      (izq.) nº de sensores informativos vs. descartados por subdataset
      (der.) distribución de vida útil (boxplot) por subdataset
    Ambos coloreados por nº de condiciones operacionales (1 vs. 6), para
    hacer visible que más condiciones -> más sensores informativos y
    -> vidas útiles más dispersas (fallos superpuestos a variabilidad
    operacional).
    """
    subsets = summary_df.index.tolist()
    cond_colors = {1: PALETTE["primary"], 6: PALETTE["danger"]}
    colors = [cond_colors[summary_df.loc[s, "Cond. operacionales"]] for s in subsets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel izquierdo: sensores informativos vs. descartados
    informative = summary_df["Sensores informativos"].values
    discarded = summary_df["Sensores descartados"].values
    x = np.arange(len(subsets))
    axes[0].bar(x, informative, color=colors, edgecolor="white", linewidth=0.5,
                label="Informativos")
    axes[0].bar(x, discarded, bottom=informative, color="#CBD5E1",
                edgecolor="white", linewidth=0.5, label="Descartados (varianza ~0)")
    for i, (inf, dis) in enumerate(zip(informative, discarded)):
        axes[0].text(i, inf / 2, str(inf), ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets)
    axes[0].set_ylabel("Nº de sensores (de 21)")
    axes[0].set_title("Sensores informativos por subdataset\n"
                       "(azul = 1 condición operacional, rojo = 6 condiciones)",
                       fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.4)

    # Panel derecho: distribución de vida útil
    life_data = [loaded[s]["train"].groupby("engine_id")["cycle"].max().values for s in subsets]
    bp = axes[1].boxplot(life_data, labels=subsets, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    axes[1].set_ylabel("Vida útil (ciclos hasta fallo)")
    axes[1].set_title("Distribución de vida útil por subdataset (Train)",
                       fontsize=11)
    axes[1].grid(axis="y", alpha=0.4)

    fig.suptitle("Comparativa de los 4 subdatasets NASA C-MAPSS",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_operating_conditions(train_df: pd.DataFrame, kmeans, subset: str,
                               save_path: str = None):
    """
    Dispersión de op_setting_1 vs. op_setting_2, coloreada por el
    régimen (k-means, ajustado en Fase 0) al que se asignó cada ciclo.
    Justifica visualmente por qué FD002/FD004 no pueden normalizarse
    con un único scaler global: los 6 clusters son claramente
    separables, es decir, los sensores se mueven en rangos distintos
    según el régimen operacional, no solo por degradación.
    """
    conditions = f0.assign_conditions(train_df, kmeans)
    n_conditions = len(np.unique(conditions))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for cond_id in sorted(conditions.unique()):
        mask = conditions == cond_id
        ax.scatter(train_df.loc[mask, "os1"], train_df.loc[mask, "os2"],
                   s=6, alpha=0.5, color=cmap(cond_id % 10),
                   label=f"Régimen {cond_id} (n={mask.sum():,})")
    ax.set_xlabel("op_setting_1")
    ax.set_ylabel("op_setting_2")
    ax.set_title(f"{subset} — {n_conditions} regímenes operacionales (k-means, fit en train)\n"
                 "Cada régimen se normaliza con su propio MinMaxScaler",
                 fontsize=11)
    ax.legend(fontsize=7, loc="best", markerscale=2)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def run_fase1_multi_subset():
    """
    Sección adicional de Fase 1: inspección comparada de los 4
    subdatasets (FD001-FD004). Se ejecuta después de run_fase1() (que
    mantiene su análisis en profundidad de FD001) y no modifica ninguno
    de sus resultados ni artefactos.
    """
    print("\n" + "═" * 70)
    print("  FASE 1 (extendida) — COMPARATIVA DE LOS 4 SUBDATASETS")
    print("═" * 70)

    print("\n[1/3] Cargando y normalizando los 4 subdatasets "
          "(estrategia según nº de condiciones operacionales)...")
    summary_df, loaded = build_subsets_comparison()
    print("\n" + summary_df.to_string())

    print("\n[2/3] Guardando tabla comparativa...")
    summary_df.to_csv(f"{OUTPUT_DIR}/fase1_subsets_comparison.csv")
    print(f"      Guardado: {OUTPUT_DIR}/fase1_subsets_comparison.csv")

    print("\n[3/3] Generando visualizaciones comparativas...")
    plot_subsets_comparison(summary_df, loaded,
                             save_path=f"{FIG_DIR}/f1_05_subsets_comparison.png")
    print(f"      Guardado: {FIG_DIR}/f1_05_subsets_comparison.png")

    # FD004 combina 6 condiciones Y 2 modos de fallo — el caso más
    # exigente del dataset, así que es el más informativo para mostrar
    # por qué hace falta normalizar por régimen en vez de globalmente.
    fd004_kmeans = loaded["FD004"]["norm_info"]["kmeans"]
    plot_operating_conditions(loaded["FD004"]["train"], fd004_kmeans, "FD004",
                               save_path=f"{FIG_DIR}/f1_06_operating_conditions_fd004.png")
    print(f"      Guardado: {FIG_DIR}/f1_06_operating_conditions_fd004.png")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 1 (EXTENDIDA)")
    print("═" * 70)
    print(f"  Subdatasets procesados : {len(summary_df)}")
    print(f"  FD001/FD003 (1 cond.)  : normalización global (idéntica a Fase 0 base)")
    print(f"  FD002/FD004 (6 cond.)  : normalización por régimen k-means")
    print(f"  Figuras generadas      : 2")
    print("═" * 70)
    print("  ✅ FASE 1 (EXTENDIDA) COMPLETADA")
    print("═" * 70 + "\n")

    return summary_df, loaded


# ─────────────────────────────────────────────
# 3. PIPELINE PRINCIPAL — FASE 1
# ─────────────────────────────────────────────

def run_fase1():
    print("\n" + "═" * 70)
    print("  FASE 1 — INSPECCIÓN Y ANÁLISIS EXPLORATORIO (sobre Fase 0)")
    print("═" * 70)

    print("\n[1/5] Cargando datos crudos (mismo loader que Fase 0)...")
    train_df, test_df, rul_df = f0.load_cmapss("FD001")

    print("\n[2/5] Inspección descriptiva...")
    sensor_stats = inspect_dataset(train_df, test_df, rul_df)
    print(sensor_stats.round(4).to_string())

    print("\n[3/5] Sensores informativos (recalculado, debe coincidir con Fase 0)...")
    informative, low_var = f0.select_informative_sensors(train_df)
    print(f"      Informativos ({len(informative)}): {informative}")
    print(f"      Descartados  ({len(low_var)}): {low_var}")

    print("\n[4/5] Ajustando/cargando escalador global (Fase 0) para la demo de señales...")
    scaler_path = f"{f0.MODEL_DIR}/global_sensor_scaler.pkl"
    if os.path.exists(scaler_path):
        import joblib
        global_scaler = joblib.load(scaler_path)
    else:
        global_scaler = f0.fit_global_scaler(train_df, informative)

    print("\n[5/5] Generando visualizaciones...")
    plot_sensor_variance(train_df, low_var, save_path=f"{FIG_DIR}/f1_01_sensor_variance.png")
    plot_life_distribution(train_df, save_path=f"{FIG_DIR}/f1_02_life_distribution.png")
    plot_sensor_distributions(train_df, informative, save_path=f"{FIG_DIR}/f1_03_sensor_distributions.png")
    plot_global_normalized_sample(train_df, informative, global_scaler,
                                   save_path=f"{FIG_DIR}/f1_04_normalized_signals.png")

    sensor_stats.to_csv(f"{OUTPUT_DIR}/sensor_stats.csv")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 1")
    print("═" * 70)
    print(f"  Motores (train/test)     : {train_df['engine_id'].nunique()} / {test_df['engine_id'].nunique()}")
    life = train_df.groupby("engine_id")["cycle"].max()
    print(f"  Vida útil media ± std    : {life.mean():.1f} ± {life.std():.1f} ciclos")
    print(f"  Sensores informativos    : {len(informative)}  (coinciden con Fase 0: "
          f"{informative == f0.INFORMATIVE_EXPECTED})")
    print(f"  Figuras generadas        : 4")
    print("═" * 70)
    print("  ✅ FASE 1 COMPLETADA")
    print("═" * 70 + "\n")

    return train_df, test_df, rul_df, sensor_stats, informative, low_var


if __name__ == "__main__":
    run_fase1()
    run_fase1_multi_subset()
