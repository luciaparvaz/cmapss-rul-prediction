"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 1: Carga, Inspección y Preprocesamiento
=============================================================================
Referencia: Saxena et al. (2008) — "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO.

Dataset: FD001 — 1 condición operacional (nivel del mar), 1 modo de fallo
(degradación del compresor de alta presión, HPC).
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
DATA_DIR    = "/home/claude/cmapss/data"
FIG_DIR     = "/home/claude/cmapss/figures"
OUTPUT_DIR  = "/home/claude/cmapss/outputs"

os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paleta de colores consistente con el proyecto
PALETTE = {
    "primary":    "#2563EB",   # azul
    "secondary":  "#10B981",   # verde
    "accent":     "#F59E0B",   # ámbar
    "danger":     "#EF4444",   # rojo
    "neutral":    "#6B7280",   # gris
    "background": "#F8FAFC",
    "text":       "#1E293B",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["background"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CBD5E1",
    "axes.labelcolor":   PALETTE["text"],
    "text.color":        PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "grid.color":        "#E2E8F0",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

# ─────────────────────────────────────────────
# 1. NOMBRES DE COLUMNAS
# ─────────────────────────────────────────────
# Según el readme oficial: 26 columnas (sin encabezado)
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"os{i}" for i in range(1, 4)]       # 3 ajustes operacionales
    + [f"s{i}"  for i in range(1, 22)]      # 21 sensores
)

SENSOR_NAMES = {
    "s1":  "T2 — Temp. inlet fan",
    "s2":  "T24 — Temp. LPC outlet",
    "s3":  "T30 — Temp. HPC outlet",
    "s4":  "T50 — Temp. LPT outlet",
    "s5":  "P2 — Pressure fan inlet",
    "s6":  "P15 — Pressure bypass",
    "s7":  "P30 — Pressure HPC outlet",
    "s8":  "Nf — Fan speed (physical)",
    "s9":  "Nc — Core speed (physical)",
    "s10": "epr — Engine pressure ratio",
    "s11": "Ps30 — Static pressure HPC",
    "s12": "phi — Fuel flow ratio",
    "s13": "NRf — Corrected fan speed",
    "s14": "NRc — Corrected core speed",
    "s15": "BPR — Bypass ratio",
    "s16": "farB — Burner fuel-air ratio",
    "s17": "htBleed — Bleed enthalpy",
    "s18": "Nf_dmd — Demanded fan speed",
    "s19": "PCNfR_dmd — Demanded corrected fan speed",
    "s20": "W31 — HPT coolant bleed",
    "s21": "W32 — LPT coolant bleed",
}

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OS_COLS     = ["os1", "os2", "os3"]


# ─────────────────────────────────────────────
# 2. FUNCIONES DE CARGA
# ─────────────────────────────────────────────

def load_cmapss(subset: str = "FD001", data_dir: str = DATA_DIR) -> tuple:
    """
    Carga los archivos train, test y RUL del subdataset indicado.

    Parámetros
    ----------
    subset   : str  — identificador del subdataset ("FD001"..FD004")
    data_dir : str  — directorio raíz donde están los .txt

    Retorna
    -------
    train_df, test_df, rul_df : DataFrames de pandas
    """
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path  = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{subset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLUMNS)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])

    # Tipos correctos
    train_df["engine_id"] = train_df["engine_id"].astype(int)
    train_df["cycle"]     = train_df["cycle"].astype(int)
    test_df["engine_id"]  = test_df["engine_id"].astype(int)
    test_df["cycle"]      = test_df["cycle"].astype(int)

    return train_df, test_df, rul_df


# ─────────────────────────────────────────────
# 3. INSPECCIÓN BÁSICA
# ─────────────────────────────────────────────

def inspect_dataset(train_df: pd.DataFrame,
                    test_df:  pd.DataFrame,
                    rul_df:   pd.DataFrame) -> pd.DataFrame:
    """
    Imprime estadísticas generales del dataset y retorna un resumen
    de sensores (media, std, min, max, varianza) para el train set.
    """
    print("=" * 70)
    print("  INSPECCIÓN DEL DATASET — NASA C-MAPSS FD001")
    print("=" * 70)

    # Dimensiones
    print(f"\n{'DIMENSIONES':─<50}")
    print(f"  Train : {train_df.shape[0]:>8,} filas  ×  {train_df.shape[1]} columnas")
    print(f"  Test  : {test_df.shape[0]:>8,} filas  ×  {test_df.shape[1]} columnas")
    print(f"  RUL   : {rul_df.shape[0]:>8,} valores")

    # Motores
    n_train_eng = train_df["engine_id"].nunique()
    n_test_eng  = test_df["engine_id"].nunique()
    print(f"\n{'MOTORES':─<50}")
    print(f"  Train : {n_train_eng} motores")
    print(f"  Test  : {n_test_eng} motores")

    # Vida útil por motor (train)
    life = train_df.groupby("engine_id")["cycle"].max()
    print(f"\n{'VIDA ÚTIL POR MOTOR (Train)':─<50}")
    print(f"  Mínima    : {life.min():>6} ciclos")
    print(f"  Máxima    : {life.max():>6} ciclos")
    print(f"  Media     : {life.mean():>9.1f} ciclos")
    print(f"  Mediana   : {life.median():>6.0f} ciclos")
    print(f"  Std       : {life.std():>9.1f} ciclos")

    # Valores nulos
    print(f"\n{'VALORES NULOS':─<50}")
    nulos_train = train_df.isnull().sum().sum()
    nulos_test  = test_df.isnull().sum().sum()
    print(f"  Train : {nulos_train} valores nulos → {'✓ Ninguno' if nulos_train == 0 else '⚠ Revisar'}")
    print(f"  Test  : {nulos_test} valores nulos → {'✓ Ninguno' if nulos_test  == 0 else '⚠ Revisar'}")

    # Resumen estadístico de sensores
    sensor_stats = train_df[SENSOR_COLS].agg(["mean", "std", "min", "max"]).T
    sensor_stats.columns = ["Media", "Std", "Mín", "Máx"]
    sensor_stats["Varianza"] = sensor_stats["Std"] ** 2
    sensor_stats["CV_%"]     = (sensor_stats["Std"] / sensor_stats["Media"].abs().replace(0, np.nan) * 100).round(2)
    sensor_stats.index.name  = "Sensor"

    print(f"\n{'RESUMEN ESTADÍSTICO DE SENSORES (Train)':─<50}")
    print(sensor_stats.round(4).to_string())

    return sensor_stats


# ─────────────────────────────────────────────
# 4. ANÁLISIS DE VARIANZA — SENSORES INFORMATIVOS
# ─────────────────────────────────────────────

def analyze_sensor_variance(train_df: pd.DataFrame,
                            std_threshold: float = 0.01) -> tuple:
    """
    Detecta sensores con varianza nula o casi nula (no informativos).

    Criterio técnico: un sensor con desviación típica < threshold
    no aporta señal de degradación y aumenta el ruido en el modelo
    (cf. Ramasso & Saxena, 2014 — "Performance Benchmarking and
     Analysis of Prognostic Methods for CMAPSS Datasets").

    Parámetros
    ----------
    std_threshold : float — umbral mínimo de std para considerar un
                            sensor informativo (default = 0.01)

    Retorna
    -------
    informative : list[str]   — sensores a conservar
    low_var     : list[str]   — sensores a descartar
    """
    stds = train_df[SENSOR_COLS].std()

    low_var     = stds[stds < std_threshold].index.tolist()
    informative = stds[stds >= std_threshold].index.tolist()

    print(f"\n{'ANÁLISIS DE VARIANZA':─<50}")
    print(f"  Umbral std  : {std_threshold}")
    print(f"  Sensores NO informativos ({len(low_var)})  : {low_var}")
    print(f"  Sensores INFORMATIVOS    ({len(informative)}) : {informative}")
    print()
    for s in SENSOR_COLS:
        flag = "✗ DESCARTAR" if s in low_var else "✓ conservar"
        print(f"    {s:4s} | std = {stds[s]:>10.4f} | {flag}  — {SENSOR_NAMES[s]}")

    return informative, low_var


# ─────────────────────────────────────────────
# 5. NORMALIZACIÓN POR MOTOR
# ─────────────────────────────────────────────

def normalize_per_unit(df: pd.DataFrame,
                       feature_cols: list,
                       method: str = "minmax") -> pd.DataFrame:
    """
    Normaliza cada feature dentro de cada motor individualmente.

    Justificación técnica: normalizar globalmente introduce el efecto
    de las condiciones iniciales (desgaste inicial variable entre motores).
    La normalización por motor elimina ese sesgo y permite al modelo
    centrarse en la tendencia intra-motor de degradación.

    FD001 usa una única condición operacional, por lo que MinMaxScaler
    por motor es suficiente. En FD002/FD004 (6 condiciones) se debería
    normalizar además por cluster de condición operacional.

    Parámetros
    ----------
    df           : DataFrame con columna engine_id
    feature_cols : columnas a normalizar
    method       : "minmax" (default) | "zscore"
    """
    df_norm = df.copy()
    # Convertir columnas de features a float64 para admitir valores escalados
    df_norm[feature_cols] = df_norm[feature_cols].astype(np.float64)

    for engine_id, group in df.groupby("engine_id"):
        idx = group.index
        vals = group[feature_cols].values

        if method == "minmax":
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(vals)
        else:  # zscore
            mean = vals.mean(axis=0)
            std  = vals.std(axis=0)
            std  = np.where(std == 0, 1, std)   # evitar división por cero
            scaled = (vals - mean) / std

        df_norm.loc[idx, feature_cols] = scaled

    return df_norm


# ─────────────────────────────────────────────
# 6. VISUALIZACIONES FASE 1
# ─────────────────────────────────────────────

def plot_sensor_variance(train_df: pd.DataFrame,
                         low_var: list,
                         save_path: str = None):
    """
    Gráfico de barras con std por sensor, marcando los no informativos.
    """
    stds = train_df[SENSOR_COLS].std().sort_values(ascending=True)

    colors = [PALETTE["danger"] if s in low_var else PALETTE["primary"]
              for s in stds.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(stds.index, stds.values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    # Línea de umbral
    ax.axvline(0.01, color=PALETTE["danger"], linestyle="--",
               linewidth=1.5, label="Umbral std = 0.01")

    # Anotaciones de valor
    for bar, val in zip(bars, stds.values):
        ax.text(val + stds.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=PALETTE["text"])

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PALETTE["primary"], label="Informativo"),
        Patch(facecolor=PALETTE["danger"],  label="No informativo (std < 0.01)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    ax.set_xlabel("Desviación Típica (std)", fontsize=11)
    ax.set_title("Variabilidad de sensores — FD001 (Train set)\n"
                 "Los sensores con std < 0.01 se descartan por ser no informativos",
                 fontsize=12, pad=15)
    ax.grid(axis="x", alpha=0.5)
    ax.set_xlim(left=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_life_distribution(train_df: pd.DataFrame, save_path: str = None):
    """
    Histograma + KDE de la distribución de vida útil de los motores.
    """
    life = train_df.groupby("engine_id")["cycle"].max()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(life.values, bins=20, color=PALETTE["primary"], alpha=0.75,
            edgecolor="white", linewidth=0.8, density=True, label="Histograma")

    # KDE manual con numpy
    from scipy.stats import gaussian_kde  # tipo check
    try:
        kde = gaussian_kde(life.values)
        x_range = np.linspace(life.min() - 10, life.max() + 10, 300)
        ax.plot(x_range, kde(x_range), color=PALETTE["danger"],
                linewidth=2.5, label="KDE")
    except ImportError:
        pass

    ax.axvline(life.mean(),   color=PALETTE["accent"],    linestyle="--",
               linewidth=2, label=f"Media  = {life.mean():.0f} ciclos")
    ax.axvline(life.median(), color=PALETTE["secondary"], linestyle=":",
               linewidth=2, label=f"Mediana = {life.median():.0f} ciclos")

    ax.set_xlabel("Vida útil (ciclos hasta fallo)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución de vida útil de los motores — FD001 (Train)\n"
                 "100 motores | 1 condición operacional | Fallo: degradación HPC",
                 fontsize=12, pad=15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    # Stats box
    stats_text = (f"n = {len(life)}  |  min = {life.min()}"
                  f"  |  máx = {life.max()}  |  std = {life.std():.1f}")
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
            ha="center", fontsize=8.5, color=PALETTE["neutral"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_sensor_distributions(train_df: pd.DataFrame,
                              informative_sensors: list,
                              save_path: str = None):
    """
    Grid de distribuciones (histograma) para cada sensor informativo.
    """
    n = len(informative_sensors)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes_flat = axes.flatten()

    for i, sensor in enumerate(informative_sensors):
        ax = axes_flat[i]
        vals = train_df[sensor].dropna()
        ax.hist(vals, bins=40, color=PALETTE["primary"], alpha=0.8,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{sensor}\n{SENSOR_NAMES[sensor]}", fontsize=8,
                     pad=4, color=PALETTE["text"])
        ax.set_xlabel("Valor", fontsize=7)
        ax.set_ylabel("Frecuencia", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Distribuciones de sensores informativos — FD001 (Train set)",
                 fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_normalized_sample(train_norm: pd.DataFrame,
                           informative_sensors: list,
                           n_motors: int = 5,
                           save_path: str = None):
    """
    Muestra la señal normalizada de 5 motores para 3 sensores clave,
    para verificar visualmente que la normalización por motor es correcta.
    """
    sensors_to_plot = ["s2", "s3", "s4"]   # sensores con degradación clara
    # filtrar solo los que son informativos
    sensors_to_plot = [s for s in sensors_to_plot if s in informative_sensors]
    if not sensors_to_plot:
        sensors_to_plot = informative_sensors[:3]

    sample_engines = train_norm["engine_id"].unique()[:n_motors]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["danger"], PALETTE["neutral"]]

    fig, axes = plt.subplots(1, len(sensors_to_plot),
                             figsize=(14, 4), sharey=False)
    if len(sensors_to_plot) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensors_to_plot):
        for j, eng in enumerate(sample_engines):
            sub = train_norm[train_norm["engine_id"] == eng]
            ax.plot(sub["cycle"], sub[sensor], color=colors[j],
                    alpha=0.85, linewidth=1.2, label=f"Motor {eng}")
        ax.set_title(f"{sensor} — {SENSOR_NAMES[sensor]}", fontsize=9)
        ax.set_xlabel("Ciclo de operación", fontsize=8)
        ax.set_ylabel("Valor normalizado [0, 1]", fontsize=8)
        ax.grid(alpha=0.35)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Señales normalizadas por motor (MinMaxScaler per-unit)\n"
                 "Se aprecia la tendencia de degradación intra-motor",
                 fontsize=11, y=1.02, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL — FASE 1
# ─────────────────────────────────────────────

def run_fase1():
    """
    Orquesta la Fase 1 completa: carga → inspección → varianza →
    normalización → exportación.
    """
    print("\n" + "═" * 70)
    print("  FASE 1 — CARGA, INSPECCIÓN Y PREPROCESAMIENTO")
    print("  NASA C-MAPSS | Subdataset FD001")
    print("═" * 70)

    # — 1. Carga —
    print("\n[1/5] Cargando archivos raw...")
    train_df, test_df, rul_df = load_cmapss("FD001")
    print(f"      Train  : {train_df.shape}  |  Test : {test_df.shape}"
          f"  |  RUL : {rul_df.shape}")

    # — 2. Inspección —
    print("\n[2/5] Inspeccionando el dataset...")
    sensor_stats = inspect_dataset(train_df, test_df, rul_df)

    # — 3. Análisis de varianza —
    print("\n[3/5] Analizando varianza de sensores...")
    informative, low_var = analyze_sensor_variance(train_df, std_threshold=0.01)

    # — 4. Normalización —
    print("\n[4/5] Normalizando datos por motor (MinMaxScaler)...")
    train_norm = normalize_per_unit(train_df, informative, method="minmax")
    test_norm  = normalize_per_unit(test_df,  informative, method="minmax")
    print(f"      Train normalizado : {train_norm.shape}")
    print(f"      Test normalizado  : {test_norm.shape}")

    # — 5. Visualizaciones —
    print("\n[5/5] Generando visualizaciones...")
    plot_sensor_variance(train_df, low_var,
        save_path=f"{FIG_DIR}/f1_01_sensor_variance.png")
    plot_life_distribution(train_df,
        save_path=f"{FIG_DIR}/f1_02_life_distribution.png")
    plot_sensor_distributions(train_df, informative,
        save_path=f"{FIG_DIR}/f1_03_sensor_distributions.png")
    plot_normalized_sample(train_norm, informative,
        save_path=f"{FIG_DIR}/f1_04_normalized_signals.png")

    # — 6. Exportar resultados —
    train_norm.to_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet", index=False)
    test_norm.to_parquet( f"{OUTPUT_DIR}/test_FD001_normalized.parquet",  index=False)
    rul_df.to_parquet(    f"{OUTPUT_DIR}/RUL_FD001.parquet",              index=False)
    sensor_stats.to_csv(  f"{OUTPUT_DIR}/sensor_stats.csv")

    print(f"\n      Archivos exportados en: {OUTPUT_DIR}/")

    # — 7. Resumen final —
    print("\n" + "═" * 70)
    print("  RESUMEN FASE 1")
    print("═" * 70)
    print(f"  Motores (train)        : {train_df['engine_id'].nunique()}")
    print(f"  Ciclos totales (train) : {len(train_df):,}")
    print(f"  Motores (test)         : {test_df['engine_id'].nunique()}")
    life = train_df.groupby("engine_id")["cycle"].max()
    print(f"  Vida útil  mín / máx  : {life.min()} / {life.max()} ciclos")
    print(f"  Vida útil  media ± std : {life.mean():.1f} ± {life.std():.1f} ciclos")
    print(f"  Valores nulos          : {train_df.isnull().sum().sum()} (train)")
    print(f"  Sensores no informativos ({len(low_var)}): {low_var}")
    print(f"  Sensores informativos  ({len(informative)}): {informative}")
    print(f"  Normalización          : MinMaxScaler por motor (intra-unit)")
    print(f"  Figuras generadas      : 4")
    print("═" * 70)
    print("  ✅ FASE 1 COMPLETADA")
    print("═" * 70 + "\n")

    return train_df, test_df, rul_df, train_norm, test_norm, informative, low_var


if __name__ == "__main__":
    results = run_fase1()
