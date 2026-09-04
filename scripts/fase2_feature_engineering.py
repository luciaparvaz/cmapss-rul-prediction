"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 2: Ingeniería de Características
=============================================================================
Referencia principal:
  - Saxena et al. (2008) — PHM08 (paper original del dataset)
  - Heimes (2008) — "Recurrent Neural Networks for Remaining Useful Life
    Estimation", PHM08: justifica el RUL cap a 125 ciclos.
  - Ramasso & Saxena (2014) — "Performance Benchmarking and Analysis of
    Prognostic Methods for CMAPSS Datasets", NASA/TM-2014-218496:
    valida ventanas de w=30 y w=50 como las más usadas en literatura.

Decisiones de diseño documentadas:
  1. RUL cap = 125 ciclos (piece-wise linear degradation)
  2. Ventanas deslizantes: w = 15, 30, 50 ciclos
  3. Features por ventana: media, std, slope (tendencia), min, max
  4. Diferencias de primer orden (Δ) para capturar velocidad de degradación
  5. Variable binaria early_failure: RUL ≤ 30 ciclos → fallo inminente
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
from scipy.stats import linregress

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DATA_DIR   = "/home/claude/cmapss/data"
OUTPUT_DIR = "/home/claude/cmapss/outputs"
FIG_DIR    = "/home/claude/cmapss/figures"

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#10B981",
    "accent":    "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#6B7280",
    "background":"#F8FAFC",
    "text":      "#1E293B",
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
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"os{i}" for i in range(1, 4)]
    + [f"s{i}"  for i in range(1, 22)]
)

# Sensores informativos identificados en Fase 1
INFORMATIVE = ['s2','s3','s4','s7','s8','s9','s11','s12',
               's13','s14','s15','s17','s20','s21']

RUL_CAP     = 125    # ciclos (Heimes 2008; Ramasso & Saxena 2014)
FAIL_THRESH = 30     # ciclos para early_failure
WINDOWS     = [15, 30, 50]


# ─────────────────────────────────────────────
# 1. CARGA DE DATOS NORMALIZADOS (Fase 1)
# ─────────────────────────────────────────────

def load_normalized():
    """
    Carga los DataFrames normalizados generados en Fase 1.
    También carga los datos raw para calcular el RUL target.
    """
    train_norm = pd.read_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet")
    test_norm  = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_normalized.parquet")
    rul_df     = pd.read_parquet(f"{OUTPUT_DIR}/RUL_FD001.parquet")
    return train_norm, test_norm, rul_df


# ─────────────────────────────────────────────
# 2. CÁLCULO DEL TARGET RUL (train)
# ─────────────────────────────────────────────

def compute_rul_target(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Calcula el RUL como target en el training set mediante:
        RUL_raw = max_cycle_por_motor − cycle_actual

    Aplica además el RUL cap (piece-wise linear degradation):
        RUL_capped = min(RUL_raw, rul_cap)

    Justificación técnica del cap:
    -----------------------------------------------------------------------
    Los motores tienen una fase de operación saludable en la que el modelo
    no tiene señal útil de degradación. Capear el RUL a 125 ciclos elimina
    esa región de "estabilidad aparente", forzando al modelo a aprender
    únicamente la zona de degradación progresiva. Heimes (2008) demuestra
    que el cap a 125 ciclos maximiza el rendimiento de los modelos
    recurrentes en FD001. Ramasso & Saxena (2014) confirman este umbral
    como consenso en la literatura de C-MAPSS.
    -----------------------------------------------------------------------

    Parámetros
    ----------
    df      : DataFrame con columnas engine_id, cycle (normalizado)
    rul_cap : int — vida útil máxima considerada (default = 125)

    Retorna
    -------
    DataFrame con columnas RUL_raw, RUL, early_failure añadidas
    """
    df = df.copy()

    # RUL crudo
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL_raw"] = max_cycle - df["cycle"]

    # RUL capeado (piece-wise linear)
    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)

    # Variable binaria de fallo inminente
    df["early_failure"] = (df["RUL_raw"] <= FAIL_THRESH).astype(int)

    return df


def compute_rul_test(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna el RUL verdadero al último ciclo de cada motor en el test set.
    Registros anteriores al último ciclo no tienen RUL asignado (NaN)
    — solo se evalúa la predicción sobre el último ciclo observado.

    Parámetros
    ----------
    test_df : DataFrame del test set (normalizado)
    rul_df  : DataFrame con columna RUL (100 valores, uno por motor)
    """
    test_df = test_df.copy()
    test_df["RUL"] = np.nan

    for eng_id in test_df["engine_id"].unique():
        mask_last = (
            (test_df["engine_id"] == eng_id) &
            (test_df["cycle"] == test_df.loc[test_df["engine_id"] == eng_id, "cycle"].max())
        )
        true_rul = rul_df.iloc[eng_id - 1]["RUL"]
        test_df.loc[mask_last, "RUL"] = true_rul

    test_df["early_failure"] = (test_df["RUL"] <= FAIL_THRESH).astype(float)
    return test_df


# ─────────────────────────────────────────────
# 3. FEATURES DE VENTANA DESLIZANTE
# ─────────────────────────────────────────────

def _rolling_slope_vectorized(y: np.ndarray, window: int) -> np.ndarray:
    """
    Calcula la pendiente de regresión lineal en ventana deslizante
    usando convolución numpy — completamente vectorizado, sin bucles Python.

    Para ventana completa (t ≥ window-1):
        slope(t) = conv(y, x_centered_flipped)[t] / x_var
    Para ventanas parciales (t < window-1):
        slope(t) se calcula sobre los puntos disponibles.

    La convolución numpy está implementada en C (BLAS/FFTW) y es
    ~100× más rápida que rolling().apply() con función Python.
    """
    n = window
    slopes = np.zeros(len(y), dtype=np.float64)

    # Kernel de convolución para ventana completa
    x_full    = np.arange(n, dtype=np.float64)
    x_mean    = x_full.mean()
    x_centered = x_full - x_mean
    x_var     = (x_centered ** 2).sum()

    if x_var > 0 and len(y) >= n:
        # Convolución de la señal con el kernel invertido → pendientes
        conv = np.convolve(y.astype(np.float64), x_centered[::-1], mode='full')
        slopes[n - 1:] = conv[n - 1:len(y)] / x_var

    # Ventanas parciales (primeros n-1 índices)
    for t in range(1, min(n - 1, len(y))):
        y_win = y[:t + 1]
        x_win = np.arange(t + 1, dtype=np.float64)
        x_m   = x_win.mean()
        x_v   = ((x_win - x_m) ** 2).sum()
        if x_v > 0:
            slopes[t] = np.dot(x_win - x_m, y_win) / x_v

    return slopes


def compute_rolling_features(df: pd.DataFrame,
                              sensor_cols: list,
                              window: int) -> pd.DataFrame:
    """
    Extrae features de ventana deslizante por motor para cada sensor.

    Features calculadas por ventana w:
      - {sensor}_mean_{w}   : media (tendencia central)
      - {sensor}_std_{w}    : desviación típica (dispersión / ruido)
      - {sensor}_slope_{w}  : pendiente OLS vectorizada (tendencia)
      - {sensor}_min_{w}    : mínimo (valor extremo inferior)
      - {sensor}_max_{w}    : máximo (valor extremo superior)

    Implementación:
      - mean/std/min/max → pandas rolling (C extension, rápido)
      - slope → convolución numpy, sin bucles Python
    """
    result_frames = []

    for eng_id, group in df.groupby("engine_id"):
        g = group.sort_values("cycle").copy()
        y_arr = g[sensor_cols].values.astype(np.float64)  # (n_ciclos, n_sensors)

        for j, col in enumerate(sensor_cols):
            series = g[col]
            prefix = f"{col}_w{window}"

            roll = series.rolling(window, min_periods=1)
            g[f"{prefix}_mean"]  = roll.mean().values
            g[f"{prefix}_std"]   = roll.std().fillna(0).values
            g[f"{prefix}_min"]   = roll.min().values
            g[f"{prefix}_max"]   = roll.max().values
            g[f"{prefix}_slope"] = _rolling_slope_vectorized(y_arr[:, j], window)

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


# ─────────────────────────────────────────────
# 4. DIFERENCIAS DE PRIMER ORDEN (Δ)
# ─────────────────────────────────────────────

def compute_first_differences(df: pd.DataFrame,
                               sensor_cols: list) -> pd.DataFrame:
    """
    Calcula la diferencia de primer orden Δs = s(t) − s(t−1) por motor.

    Justificación técnica:
    La diferencia de primer orden captura la VELOCIDAD de cambio del
    sensor, que en motores en degradación tiende a aumentar en módulo
    conforme se acerca el fallo. Es un proxy de la derivada discreta
    del proceso de degradación. Especialmente útil para s3 (T30) y s4
    (T50), cuya velocidad de cambio se acelera ~30 ciclos antes del fallo.

    El primer ciclo de cada motor recibe Δ = 0 (sin referencia anterior).
    """
    df = df.copy()

    for eng_id, group in df.groupby("engine_id"):
        idx = group.sort_values("cycle").index
        for col in sensor_cols:
            delta_col = f"{col}_delta"
            if delta_col not in df.columns:
                df[delta_col] = 0.0
            df.loc[idx, delta_col] = df.loc[idx, col].diff().fillna(0).values

    return df


# ─────────────────────────────────────────────
# 5. PIPELINE COMPLETO DE FEATURES
# ─────────────────────────────────────────────

def build_feature_set(df: pd.DataFrame,
                      sensor_cols: list = INFORMATIVE,
                      windows: list = WINDOWS,
                      include_delta: bool = True) -> pd.DataFrame:
    """
    Orquesta la construcción completa del feature set:
      1. Features rolling para cada ventana en `windows`
      2. Diferencias de primer orden (delta)

    Parámetros
    ----------
    df          : DataFrame con RUL ya calculado
    sensor_cols : sensores informativos a usar
    windows     : lista de tamaños de ventana
    include_delta: si True, añade diferencias de primer orden

    Retorna
    -------
    DataFrame enriquecido con todas las features
    """
    print(f"    Sensores base       : {len(sensor_cols)}")
    print(f"    Ventanas            : {windows}")

    # Rolling features para cada ventana
    df_out = df.copy()
    for w in windows:
        print(f"    Calculando rolling w={w}...", end=" ")
        df_out = compute_rolling_features(df_out, sensor_cols, window=w)
        n_new = len(sensor_cols) * 5   # 5 stats por sensor
        print(f"+{n_new} features")

    # Diferencias de primer orden
    if include_delta:
        print(f"    Calculando deltas...", end=" ")
        df_out = compute_first_differences(df_out, sensor_cols)
        print(f"+{len(sensor_cols)} features (Δ)")

    # Conteo total
    base_cols   = ["engine_id", "cycle"] + [c for c in df_out.columns
                    if c.startswith("os")] + ["RUL_raw", "RUL", "early_failure"]
    feature_cols = [c for c in df_out.columns if c not in base_cols
                    and c not in sensor_cols]

    print(f"    Features engineered : {len(feature_cols)}")
    print(f"    Columnas totales    : {df_out.shape[1]}")

    return df_out


# ─────────────────────────────────────────────
# 6. VISUALIZACIONES FASE 2
# ─────────────────────────────────────────────

def plot_rul_distribution(train_with_rul: pd.DataFrame,
                          save_path: str = None):
    """
    Compara la distribución del RUL raw vs. RUL capeado en el train set.
    Evidencia visual del efecto del RUL cap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title, color in zip(
        axes,
        ["RUL_raw", "RUL"],
        ["RUL crudo (sin cap)", f"RUL capeado (cap = {RUL_CAP} ciclos)"],
        [PALETTE["neutral"], PALETTE["primary"]]
    ):
        vals = train_with_rul[col]
        ax.hist(vals, bins=50, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(vals.mean(),   color=PALETTE["accent"],  linestyle="--",
                   linewidth=2, label=f"Media = {vals.mean():.1f}")
        ax.axvline(vals.median(), color=PALETTE["danger"],  linestyle=":",
                   linewidth=2, label=f"Mediana = {vals.median():.1f}")
        if col == "RUL":
            ax.axvline(FAIL_THRESH, color=PALETTE["secondary"], linestyle="-.",
                       linewidth=2, label=f"early_failure (≤{FAIL_THRESH})")
        ax.set_title(title, pad=10)
        ax.set_xlabel("RUL (ciclos)")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.35)

    fig.suptitle("Distribución del target RUL — Efecto del piece-wise linear cap\n"
                 "FD001 Train set | Cap justificado por Heimes (2008) & Ramasso-Saxena (2014)",
                 fontsize=11, y=1.02, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_early_failure_balance(train_with_rul: pd.DataFrame,
                               save_path: str = None):
    """
    Gráfico de barras con el balance de clases para early_failure.
    Muestra el desbalanceo natural del problema de clasificación.
    """
    counts = train_with_rul["early_failure"].value_counts().sort_index()
    labels = [f"Normal\n(RUL > {FAIL_THRESH})", f"Fallo inminente\n(RUL ≤ {FAIL_THRESH})"]
    colors = [PALETTE["primary"], PALETTE["danger"]]
    total  = counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Barras
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="white",
                   linewidth=0.8, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.005,
                 f"{val:,}\n({val/total*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Número de ciclos", fontsize=11)
    ax1.set_title("Balance de clases — early_failure", pad=10)
    ax1.grid(axis="y", alpha=0.4)
    ax1.set_ylim(0, counts.max() * 1.15)

    # Pie
    ax2.pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("Proporción de clases", pad=10)

    fig.suptitle(f"Balance de clases para clasificación binaria\n"
                 f"Umbral early_failure: RUL ≤ {FAIL_THRESH} ciclos",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_rolling_features_example(train_features: pd.DataFrame,
                                  save_path: str = None):
    """
    Muestra las features rolling (mean, std, slope) de s3 (T30 HPC)
    para 3 motores representativos, con el RUL superpuesto.
    Visualiza cómo cada feature captura distintos aspectos de la degradación.
    """
    sample_engines = [1, 5, 10]
    sensor = "s3"
    windows_to_plot = [15, 30, 50]
    feature_types = ["mean", "std", "slope"]
    feat_colors    = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    motor_styles   = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, feat_type, fc in zip(axes, feature_types, feat_colors):
        for eng, lstyle in zip(sample_engines, motor_styles):
            sub = train_features[train_features["engine_id"] == eng].sort_values("cycle")
            for w in windows_to_plot:
                col = f"{sensor}_w{w}_{feat_type}"
                if col in sub.columns:
                    alpha = {15: 0.55, 30: 0.80, 50: 1.0}[w]
                    lw    = {15: 1.0,  30: 1.4,  50: 2.0}[w]
                    ax.plot(sub["RUL"], sub[col],
                            color=fc, alpha=alpha, linewidth=lw,
                            linestyle=lstyle,
                            label=f"Motor {eng}, w={w}" if w == 30 else "")

        ax.axvline(FAIL_THRESH, color=PALETTE["danger"], linestyle="--",
                   linewidth=1.5, alpha=0.8, label=f"Zona fallo (≤{FAIL_THRESH})")
        ax.invert_xaxis()   # RUL decreciente → fallo a la derecha
        ax.set_xlabel("RUL (ciclos restantes)", fontsize=9)
        ax.set_title(f"s3 — {feat_type.upper()}\n(w = 15, 30, 50 ciclos)", fontsize=10)
        ax.grid(alpha=0.35)

        # Leyenda solo en primer panel
        if feat_type == "mean":
            ax.set_ylabel("Valor normalizado", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Features de ventana deslizante — Sensor s3 (T30 Temp. HPC outlet)\n"
                 "Eje X invertido: izquierda = inicio de vida, derecha = fallo inminente",
                 fontsize=11, y=1.02, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


def plot_delta_features(train_features: pd.DataFrame,
                        save_path: str = None):
    """
    Muestra las diferencias de primer orden (Δ) de 4 sensores clave
    para evidenciar cómo la velocidad de degradación aumenta hacia el fallo.
    """
    sensors_to_show = ["s3", "s4", "s14", "s17"]
    sensor_labels   = {
        "s3": "s3 — T30 (HPC outlet temp)",
        "s4": "s4 — T50 (LPT outlet temp)",
        "s14":"s14 — NRc (corrected core speed)",
        "s17":"s17 — htBleed (bleed enthalpy)",
    }
    sample_engines = [1, 5, 10, 15]
    colors = [PALETTE["primary"], PALETTE["secondary"],
              PALETTE["accent"], PALETTE["danger"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, sensor in zip(axes_flat, sensors_to_show):
        delta_col = f"{sensor}_delta"
        for eng, col in zip(sample_engines, colors):
            sub = train_features[train_features["engine_id"] == eng].sort_values("cycle")
            if delta_col in sub.columns:
                ax.plot(sub["RUL"], sub[delta_col].rolling(5).mean(),
                        color=col, alpha=0.8, linewidth=1.3,
                        label=f"Motor {eng}")
        ax.axvline(FAIL_THRESH, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8, alpha=0.5)
        ax.invert_xaxis()
        ax.set_title(sensor_labels[sensor], fontsize=10)
        ax.set_xlabel("RUL (ciclos restantes)", fontsize=8)
        ax.set_ylabel("Δ (diff. orden 1, suavizada)", fontsize=8)
        ax.grid(alpha=0.35)
        ax.legend(fontsize=7)

    fig.suptitle("Diferencias de primer orden (Δ) — Velocidad de degradación por sensor\n"
                 "Eje X invertido | Línea negra = umbral early_failure (RUL=30)",
                 fontsize=11, y=1.01, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL — FASE 2
# ─────────────────────────────────────────────

def run_fase2():
    print("\n" + "═" * 70)
    print("  FASE 2 — INGENIERÍA DE CARACTERÍSTICAS")
    print("  NASA C-MAPSS | Subdataset FD001")
    print("═" * 70)

    # — 1. Cargar datos normalizados —
    print("\n[1/6] Cargando datos normalizados (Fase 1)...")
    train_norm, test_norm, rul_df = load_normalized()
    print(f"      Train: {train_norm.shape}  |  Test: {test_norm.shape}")

    # — 2. Calcular targets RUL —
    print("\n[2/6] Calculando target RUL (train) con cap = {RUL_CAP} ciclos...")
    train_rul = compute_rul_target(train_norm, rul_cap=RUL_CAP)
    test_rul  = compute_rul_test(test_norm, rul_df)

    # Estadísticas del target
    print(f"      RUL_raw — media: {train_rul['RUL_raw'].mean():.1f} "
          f"| máx: {train_rul['RUL_raw'].max()} ciclos")
    print(f"      RUL_cap — media: {train_rul['RUL'].mean():.1f} "
          f"| máx: {train_rul['RUL'].max()} ciclos")
    ef_pct = train_rul["early_failure"].mean() * 100
    print(f"      early_failure — {ef_pct:.1f}% de ciclos en zona de riesgo "
          f"(RUL ≤ {FAIL_THRESH})")

    # — 3. Features de ventana deslizante —
    print(f"\n[3/6] Construyendo features de ventana deslizante (w = {WINDOWS})...")
    train_features = build_feature_set(train_rul, INFORMATIVE, WINDOWS)
    test_features  = build_feature_set(
        test_rul.fillna({"RUL": 0, "early_failure": 0}),
        INFORMATIVE, WINDOWS
    )

    # — 4. Resumen del feature set —
    base_meta = ["engine_id", "cycle", "os1", "os2", "os3",
                 "RUL_raw", "RUL", "early_failure"]
    feature_cols = [c for c in train_features.columns
                    if c not in base_meta and c not in INFORMATIVE]
    print(f"\n[4/6] Resumen del feature set:")
    print(f"      Sensores base      : {len(INFORMATIVE)}")
    delta_cols   = [c for c in feature_cols if "_delta" in c]
    rolling_cols = [c for c in feature_cols if "_w"    in c]
    print(f"      Features rolling   : {len(rolling_cols)}  "
          f"({len(INFORMATIVE)} sensores × {len(WINDOWS)} ventanas × 5 stats)")
    print(f"      Features delta     : {len(delta_cols)}  "
          f"({len(INFORMATIVE)} sensores × Δ orden 1)")
    print(f"      TOTAL features ML  : {len(feature_cols) + len(INFORMATIVE)}")
    print(f"      Columnas totales   : {train_features.shape[1]}")

    # — 5. Visualizaciones —
    print("\n[5/6] Generando visualizaciones...")
    plot_rul_distribution(train_rul,
        save_path=f"{FIG_DIR}/f2_01_rul_distribution.png")
    plot_early_failure_balance(train_rul,
        save_path=f"{FIG_DIR}/f2_02_early_failure_balance.png")
    plot_rolling_features_example(train_features,
        save_path=f"{FIG_DIR}/f2_03_rolling_features.png")
    plot_delta_features(train_features,
        save_path=f"{FIG_DIR}/f2_04_delta_features.png")

    # — 6. Exportar —
    print("\n[6/6] Exportando datasets con features...")
    train_features.to_parquet(f"{OUTPUT_DIR}/train_FD001_features.parquet", index=False)
    test_features.to_parquet( f"{OUTPUT_DIR}/test_FD001_features.parquet",  index=False)
    print(f"      train_features : {train_features.shape}")
    print(f"      test_features  : {test_features.shape}")

    # — Resumen —
    print("\n" + "═" * 70)
    print("  RESUMEN FASE 2")
    print("═" * 70)
    print(f"  RUL cap aplicado        : {RUL_CAP} ciclos (Heimes 2008)")
    print(f"  Umbral early_failure    : {FAIL_THRESH} ciclos")
    print(f"  % ciclos zona de riesgo : {ef_pct:.1f}%")
    print(f"  Ventanas deslizantes    : {WINDOWS} ciclos")
    print(f"  Features rolling        : {len(rolling_cols)} ({len(INFORMATIVE)}×{len(WINDOWS)}×5)")
    print(f"  Features delta (Δ)      : {len(delta_cols)}")
    print(f"  Total features ML       : {len(feature_cols) + len(INFORMATIVE)}")
    print(f"  Dimensión train final   : {train_features.shape}")
    print(f"  Figuras generadas       : 4")
    print("═" * 70)
    print("  ✅ FASE 2 COMPLETADA")
    print("═" * 70 + "\n")

    return train_features, test_features, feature_cols


if __name__ == "__main__":
    train_features, test_features, feature_cols = run_fase2()
