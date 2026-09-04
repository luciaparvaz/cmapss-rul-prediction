"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001 / FD003)
FASE 3: Análisis Exploratorio Orientado a Degradación
=============================================================================
Objetivo: visualizar y cuantificar los patrones de deterioro en los sensores
a lo largo del ciclo de vida de los motores, identificar los sensores más
discriminativos y comparar el comportamiento entre FD001 (1 modo de fallo)
y FD003 (2 modos de fallo).

Visualizaciones producidas:
  f3_01_spaghetti_plots.png     — Trayectorias de degradación por sensor
  f3_02_correlation_heatmap.png — Correlación sensores × sensores × RUL
  f3_03_rul_by_life_quartile.png— Distribución del RUL por cuartil de vida
  f3_04_degradation_phases.png  — Análisis de fases de degradación
  f3_05_fd001_vs_fd003.png      — Comparativa FD001 vs FD003

Referencias:
  - Saxena et al. (2008) — Paper original C-MAPSS, PHM08
  - Ramasso & Saxena (2014) — NASA/TM-2014-218496
  - Li et al. (2018) — Reliability Eng. & System Safety, 172, 1-11
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
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DATA_DIR   = "/home/claude/cmapss/data"
OUTPUT_DIR = "/home/claude/cmapss/outputs"
FIG_DIR    = "/home/claude/cmapss/figures"

COLUMNS = (["engine_id","cycle"]
           + [f"os{i}" for i in range(1,4)]
           + [f"s{i}"  for i in range(1,22)])

INFORMATIVE = ['s2','s3','s4','s7','s8','s9','s11','s12',
               's13','s14','s15','s17','s20','s21']

SENSOR_LABELS = {
    's2':  'T24 — Temp. LPC outlet',
    's3':  'T30 — Temp. HPC outlet',
    's4':  'T50 — Temp. LPT outlet',
    's7':  'P30 — Pressure HPC outlet',
    's8':  'Nf — Fan speed',
    's9':  'Nc — Core speed',
    's11': 'Ps30 — Static pressure HPC',
    's12': 'phi — Fuel flow ratio',
    's13': 'NRf — Corrected fan speed',
    's14': 'NRc — Corrected core speed',
    's15': 'BPR — Bypass ratio',
    's17': 'htBleed — Bleed enthalpy',
    's20': 'W31 — HPT coolant bleed',
    's21': 'W32 — LPT coolant bleed',
}

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#10B981",
    "accent":    "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#6B7280",
    "background":"#F8FAFC",
    "text":      "#1E293B",
    "purple":    "#7C3AED",
    "teal":      "#0D9488",
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
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

RUL_CAP = 125


# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────

def load_raw(subset):
    df = pd.read_csv(f"{DATA_DIR}/train_{subset}.txt",
                     sep=r"\s+", header=None, names=COLUMNS)
    df["rul_raw"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    df["RUL"]     = df["rul_raw"].clip(upper=RUL_CAP)
    df["early_failure"] = (df["rul_raw"] <= 30).astype(int)
    df["life_pct"] = df["cycle"] / df.groupby("engine_id")["cycle"].transform("max")
    return df


# ─────────────────────────────────────────────
# FIGURA 1 — SPAGHETTI PLOTS
# Trayectorias de degradación por sensor (muestra de 20 motores)
# ─────────────────────────────────────────────

def plot_spaghetti(train_df, save_path=None):
    """
    Spaghetti plot: cada línea = un motor, eje X = RUL (invertido,
    de modo que el fallo queda a la derecha).

    Se muestran los 8 sensores con mayor correlación absoluta con RUL,
    identificados en la Fase 2 (auditoría B2).

    El coloreado por RUL restante (degradé azul→rojo) permite identificar
    visualmente cuándo empieza la señal de degradación observable.
    """
    top_sensors = ['s11','s4','s12','s7','s15','s21','s20','s2']
    n_motors_plot = 20
    sample_ids = np.random.choice(
        train_df["engine_id"].unique(), n_motors_plot, replace=False
    )

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes_flat = axes.flatten()

    # Colormap continuo RUL: azul (sano) → rojo (próximo al fallo)
    cmap = plt.cm.RdYlGn   # verde = sano, rojo = fallo

    for ax, sensor in zip(axes_flat, top_sensors):
        for eng in sample_ids:
            sub = train_df[train_df["engine_id"] == eng].sort_values("cycle")
            rul_vals = sub["rul_raw"].values
            s_vals   = sub[sensor].values

            # Color por RUL normalizado [0,1]: 0=fallo, 1=sano
            norm_rul = np.clip(rul_vals / RUL_CAP, 0, 1)

            # Dibujar segmento a segmento para el gradiente de color
            for t in range(len(sub) - 1):
                color = cmap(norm_rul[t])
                ax.plot(rul_vals[t:t+2], s_vals[t:t+2],
                        color=color, alpha=0.55, linewidth=0.9)

        ax.axvline(30, color=PALETTE["danger"], linestyle="--",
                   linewidth=1.5, alpha=0.8, label="Zona fallo (RUL=30)")
        ax.axvline(RUL_CAP, color=PALETTE["neutral"], linestyle=":",
                   linewidth=1.2, alpha=0.6, label=f"Cap RUL={RUL_CAP}")
        ax.invert_xaxis()
        ax.set_xlabel("RUL (ciclos restantes)", fontsize=8)
        ax.set_ylabel("Valor del sensor", fontsize=8)
        ax.set_title(f"{sensor} — {SENSOR_LABELS[sensor]}", pad=6)
        ax.grid(alpha=0.3)

        # Correlación Pearson en el título
        r, _ = pearsonr(train_df[sensor], train_df["rul_raw"])
        ax.set_title(
            f"{sensor} — {SENSOR_LABELS[sensor]}\n"
            f"Pearson r = {r:.3f} con RUL",
            pad=5, fontsize=9
        )

    # Colorbar global
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=0, vmax=RUL_CAP))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, orientation="horizontal",
                        fraction=0.02, pad=0.03, aspect=50)
    cbar.set_label("RUL restante (ciclos) — verde: sano  |  rojo: fallo inminente",
                   fontsize=10)

    fig.suptitle(
        "Trayectorias de degradación por sensor — NASA C-MAPSS FD001\n"
        f"Muestra de {n_motors_plot} motores | Eje X invertido | "
        "Top-8 sensores por correlación con RUL",
        fontsize=12, fontweight="bold", y=1.005
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 2 — HEATMAP DE CORRELACIÓN
# Sensores × sensores × RUL (Pearson y Spearman)
# ─────────────────────────────────────────────

def plot_correlation_heatmap(train_df, save_path=None):
    """
    Dos heatmaps: correlación de Pearson (relaciones lineales) y
    de Spearman (relaciones monótonas), incluyendo RUL como variable
    adicional. El heatmap de Spearman captura correlaciones no lineales
    que Pearson podría subestimar en señales de degradación acelerada.
    """
    cols   = INFORMATIVE + ["rul_raw"]
    labels = [SENSOR_LABELS.get(c, c) if c != "rul_raw" else "RUL raw" for c in cols]

    df_sub = train_df[cols].copy()

    pearson_mat  = df_sub.corr(method="pearson")
    spearman_mat = df_sub.corr(method="spearman")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    kw = dict(
        xticklabels=labels, yticklabels=labels,
        vmin=-1, vmax=1, center=0,
        cmap="coolwarm",
        square=True, linewidths=0.4, linecolor="#E2E8F0",
        annot=True, fmt=".2f", annot_kws={"size": 6.5},
        cbar_kws={"shrink": 0.75, "label": "Coeficiente de correlación"},
    )

    sns.heatmap(pearson_mat,  ax=ax1, **kw)
    sns.heatmap(spearman_mat, ax=ax2, **kw)

    ax1.set_title("Correlación de Pearson\n(relaciones lineales)", pad=10)
    ax2.set_title("Correlación de Spearman\n(relaciones monótonas, no lineales)", pad=10)

    for ax in (ax1, ax2):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    fig.suptitle(
        "Mapa de correlación — Sensores informativos × RUL | FD001 Train set\n"
        "Última fila/columna = correlación directa de cada sensor con el RUL",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 3 — DISTRIBUCIÓN DEL RUL POR CUARTIL DE VIDA
# ─────────────────────────────────────────────

def plot_rul_by_life_quartile(train_df, save_path=None):
    """
    Divide los motores en 4 grupos según su vida total (cuartiles Q1..Q4)
    y compara la distribución de RUL cap y la trayectoria de degradación
    del sensor más correlacionado (s11) en cada grupo.

    Justificación: si el modelo no se generaliza entre motores de distinta
    vida total, se introducirá sesgo sistemático. Esta visualización permite
    detectar si la distribución del target RUL es comparable entre grupos.
    """
    life = train_df.groupby("engine_id")["cycle"].max()
    quartiles = pd.qcut(life, q=4, labels=["Q1 (corta)", "Q2", "Q3", "Q4 (larga)"])
    life_q = quartiles.reset_index()
    life_q.columns = ["engine_id", "life_quartile"]
    df_q = train_df.merge(life_q, on="engine_id")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    colors_q = [PALETTE["primary"], PALETTE["secondary"],
                PALETTE["accent"],  PALETTE["danger"]]
    quartile_labels = ["Q1 (corta)", "Q2", "Q3", "Q4 (larga)"]

    # Panel superior izquierdo: distribución de vida total por cuartil
    ax0 = fig.add_subplot(gs[0, 0])
    for q, col in zip(quartile_labels, colors_q):
        eng_ids = life_q[life_q["life_quartile"] == q]["engine_id"]
        q_lives = life[eng_ids].values
        ax0.hist(q_lives, bins=12, alpha=0.65, color=col, label=q,
                 edgecolor="white", linewidth=0.5)
    ax0.set_xlabel("Vida total (ciclos)"); ax0.set_ylabel("N.º de motores")
    ax0.set_title("Distribución de vida total por cuartil", pad=8)
    ax0.legend(fontsize=8); ax0.grid(alpha=0.35)

    # Panel superior derecho: distribución del RUL (capeado) por cuartil
    ax1 = fig.add_subplot(gs[0, 1])
    for q, col in zip(quartile_labels, colors_q):
        vals = df_q[df_q["life_quartile"] == q]["RUL"]
        ax1.hist(vals, bins=30, alpha=0.55, color=col, label=q,
                 edgecolor="white", linewidth=0.4, density=True)
    ax1.set_xlabel("RUL capeado (ciclos)"); ax1.set_ylabel("Densidad")
    ax1.set_title("Distribución del target RUL por cuartil de vida", pad=8)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.35)

    # Panel inferior: trayectorias de s11 normalizadas por vida total
    # (% de vida transcurrido en eje X para comparar motores de distinta duración)
    ax2 = fig.add_subplot(gs[1, :])
    for q, col in zip(quartile_labels, colors_q):
        eng_ids = life_q[life_q["life_quartile"] == q]["engine_id"].values[:5]
        for eng in eng_ids:
            sub = train_df[train_df["engine_id"] == eng].sort_values("cycle")
            pct = sub["cycle"] / sub["cycle"].max()
            ax2.plot(pct, sub["s11"], color=col, alpha=0.45, linewidth=1.0)

    # Una línea por cuartil para la leyenda
    for q, col in zip(quartile_labels, colors_q):
        ax2.plot([], [], color=col, linewidth=2, label=q)

    ax2.axvline(0.70, color="black", linestyle=":", linewidth=1.2,
                label="70% de vida (inicio degradación típico)")
    ax2.set_xlabel("Porcentaje de vida transcurrido (%)")
    ax2.set_ylabel("Sensor s11 — Ps30 Static pressure HPC")
    ax2.set_title(
        "Trayectorias de s11 normalizadas por vida total (% de vida)\n"
        "Permite comparar la forma de degradación entre motores de distinta duración",
        pad=8
    )
    ax2.legend(fontsize=8, loc="upper right"); ax2.grid(alpha=0.3)

    fig.suptitle(
        "Análisis del RUL y degradación por cuartil de vida útil — FD001\n"
        "¿La distribución del target es homogénea entre motores cortos y largos?",
        fontsize=12, fontweight="bold"
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 4 — ANÁLISIS DE FASES DE DEGRADACIÓN
# Mediana de sensores clave por ventana de RUL
# ─────────────────────────────────────────────

def plot_degradation_phases(train_df, save_path=None):
    """
    Agrupa todos los ciclos de todos los motores por intervalo de RUL
    y calcula la mediana del sensor en cada intervalo.

    Esto produce una "curva de degradación media" que revela:
    1. En qué momento del RUL comienza a separarse la señal del ruido.
    2. Si la degradación es lineal, exponencial o con quiebres.
    3. Qué sensores muestran el cambio más temprano (mayor lead time).
    """
    sensors_to_plot = ['s11','s4','s12','s7','s15','s3']
    bin_size = 5
    rul_bins = np.arange(0, train_df["rul_raw"].max() + bin_size, bin_size)
    train_df["rul_bin"] = pd.cut(train_df["rul_raw"], bins=rul_bins,
                                  labels=rul_bins[:-1], right=False)
    train_df["rul_bin"] = train_df["rul_bin"].astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax, sensor in zip(axes_flat, sensors_to_plot):
        grouped = train_df.groupby("rul_bin")[sensor].agg(["median","std","count"])
        grouped = grouped[grouped["count"] >= 5]   # al menos 5 motores en el bin
        rul_x   = grouped.index.values
        med     = grouped["median"].values
        s_std   = grouped["std"].values

        # Mediana
        ax.plot(rul_x, med, color=PALETTE["primary"], linewidth=2,
                label="Mediana")

        # Banda de incertidumbre (±1 std)
        ax.fill_between(rul_x, med - s_std, med + s_std,
                        color=PALETTE["primary"], alpha=0.18,
                        label="±1 std")

        # Zona de fallo
        ax.axvspan(0, 30, color=PALETTE["danger"], alpha=0.10,
                   label="Zona fallo (RUL≤30)")
        ax.axvline(30, color=PALETTE["danger"], linestyle="--",
                   linewidth=1.3, alpha=0.8)
        ax.axvline(RUL_CAP, color=PALETTE["neutral"], linestyle=":",
                   linewidth=1.1, alpha=0.6, label=f"Cap RUL={RUL_CAP}")

        # Correlación Pearson en etiqueta
        r, _ = pearsonr(train_df[sensor], train_df["rul_raw"])

        ax.set_xlabel("RUL (ciclos restantes)", fontsize=8)
        ax.set_ylabel("Valor del sensor (mediana)", fontsize=8)
        ax.set_title(
            f"{sensor} — {SENSOR_LABELS[sensor]}\n"
            f"Pearson r(sensor, RUL) = {r:.3f}",
            fontsize=9, pad=5
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        # No invertir eje X aquí: RUL creciente = motor sano a la derecha

    fig.suptitle(
        "Curvas de degradación media por intervalo de RUL — FD001 Train set\n"
        "Mediana ± 1 std sobre todos los motores | Bin size = 5 ciclos",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 5 — COMPARATIVA FD001 vs FD003
# 1 modo de fallo vs 2 modos de fallo
# ─────────────────────────────────────────────

def plot_fd001_vs_fd003(save_path=None):
    """
    FD001: 1 condición operacional, 1 modo de fallo (HPC degradation)
    FD003: 1 condición operacional, 2 modos de fallo (HPC + Fan degradation)

    Esta comparativa examina:
    1. Distribución de vida útil (¿es diferente entre datasets?)
    2. Trayectorias de sensores críticos (¿se ve el segundo modo de fallo?)
    3. Dispersión inter-motor (¿hay mayor variabilidad en FD003?)

    Justificación de la comparativa: antes de generalizar el modelo a otros
    subdatasets, es necesario verificar que la estructura de degradación
    es comparable o si requiere estrategias distintas.
    """
    # Cargar FD003 si está disponible
    fd003_path = f"{DATA_DIR}/train_FD003.txt"
    if not os.path.exists(fd003_path):
        print("  ⚠ train_FD003.txt no disponible en el workspace — omitiendo Fig. 5")
        return

    fd001 = load_raw("FD001")
    fd003 = load_raw("FD003")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {"FD001": PALETTE["primary"], "FD003": PALETTE["danger"]}

    # ── Panel (0,0): Distribución de vida total ──
    ax = axes[0, 0]
    for name, df in [("FD001", fd001), ("FD003", fd003)]:
        life = df.groupby("engine_id")["cycle"].max()
        ax.hist(life, bins=20, alpha=0.65, color=colors[name],
                edgecolor="white", linewidth=0.5, label=name, density=True)
        ax.axvline(life.mean(), color=colors[name], linestyle="--",
                   linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Vida total (ciclos)"); ax.set_ylabel("Densidad")
    ax.set_title("Distribución de vida útil\nFD001 vs FD003")
    ax.legend(fontsize=9); ax.grid(alpha=0.35)

    # ── Panel (0,1): Distribución del RUL capeado ──
    ax = axes[0, 1]
    for name, df in [("FD001", fd001), ("FD003", fd003)]:
        ax.hist(df["RUL"], bins=30, alpha=0.55, color=colors[name],
                edgecolor="white", linewidth=0.4, label=name, density=True)
    ax.set_xlabel("RUL capeado (ciclos)"); ax.set_ylabel("Densidad")
    ax.set_title("Distribución del target RUL\nFD001 vs FD003 (cap=125)")
    ax.legend(fontsize=9); ax.grid(alpha=0.35)

    # ── Panel (0,2): Tabla de estadísticas ──
    ax = axes[0, 2]
    ax.axis("off")
    stats_data = []
    for name, df in [("FD001", fd001), ("FD003", fd003)]:
        life = df.groupby("engine_id")["cycle"].max()
        stats_data.append([
            name,
            df["engine_id"].nunique(),
            f"{life.min()}",
            f"{life.max()}",
            f"{life.mean():.1f}",
            f"{life.std():.1f}",
            "1 (HPC)",
            "1" if name=="FD001" else "2 (HPC+Fan)"
        ])
    table = ax.table(
        cellText=stats_data,
        colLabels=["Dataset","Motores","Min","Max","Media","Std",
                   "Cond. op.","Modos fallo"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EFF6FF")
    ax.set_title("Estadísticas comparativas", fontweight="bold", pad=15)

    # ── Paneles (1,0..2): Trayectorias de 3 sensores clave ──
    sensors_cmp = ["s3", "s11", "s4"]
    for i, sensor in enumerate(sensors_cmp):
        ax = axes[1, i]
        sample_n = 15
        for name, df, ls in [("FD001", fd001, "-"), ("FD003", fd003, "--")]:
            sample_ids = np.random.choice(
                df["engine_id"].unique(), min(sample_n, df["engine_id"].nunique()),
                replace=False
            )
            for eng in sample_ids:
                sub = df[df["engine_id"] == eng].sort_values("cycle")
                pct = sub["cycle"] / sub["cycle"].max()
                ax.plot(pct * 100, sub[sensor],
                        color=colors[name], alpha=0.30,
                        linewidth=0.9, linestyle=ls)
            # Curva mediana del dataset
            grouped = df.assign(
                pct_bin=pd.cut(df["life_pct"]*100, bins=50,
                               labels=np.linspace(1, 100, 50))
            ).groupby("pct_bin")[sensor].median()
            ax.plot(np.linspace(1, 100, 50), grouped.values,
                    color=colors[name], linewidth=2.5, linestyle=ls,
                    label=f"{name} (mediana)")

        ax.axvline(70, color="black", linestyle=":", linewidth=1,
                   alpha=0.5, label="70% vida")
        ax.set_xlabel("Vida transcurrida (%)")
        ax.set_ylabel(f"Sensor {sensor}")
        ax.set_title(
            f"{sensor} — {SENSOR_LABELS[sensor]}\nFD001 vs FD003",
            pad=5
        )
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(
        "Comparativa FD001 vs FD003 — 1 modo de fallo (HPC) vs 2 modos (HPC + Fan)\n"
        "1 condición operacional en ambos | Eje X = % de vida transcurrida",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Guardado: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# ANÁLISIS ESTADÍSTICO TEXTUAL
# ─────────────────────────────────────────────

def print_eda_summary(train_df):
    """
    Calcula y reporta métricas estadísticas de la EDA:
    correlaciones Pearson y Spearman, punto de inicio de degradación,
    y análisis de varianza inter- vs intra-motor.
    """
    print("\n" + "─"*60)
    print("ANÁLISIS ESTADÍSTICO — FASE 3")
    print("─"*60)

    # Correlaciones con RUL
    print("\n[1] Correlación con RUL_raw (Pearson y Spearman):")
    print(f"    {'Sensor':<6} {'Pearson r':>12} {'Spearman ρ':>12} {'Señal':>10}")
    for s in INFORMATIVE:
        r_p, _ = pearsonr(train_df[s], train_df["rul_raw"])
        r_s, _ = spearmanr(train_df[s], train_df["rul_raw"])
        signal  = "ALTA" if abs(r_p) > 0.6 else ("MEDIA" if abs(r_p) > 0.3 else "BAJA")
        print(f"    {s:<6} {r_p:>12.4f} {r_s:>12.4f} {signal:>10}")

    # Punto de inicio de degradación
    print("\n[2] Punto de inicio de degradación observable (por sensor):")
    print("    (RUL a partir del cual la señal supera 2σ del valor basal)")
    baseline_mask = train_df["rul_raw"] > 200   # fase sana
    for s in ['s11','s4','s12','s7','s3']:
        base_mean = train_df.loc[baseline_mask, s].mean()
        base_std  = train_df.loc[baseline_mask, s].std()
        threshold_above = base_mean + 2 * base_std
        threshold_below = base_mean - 2 * base_std

        # Buscar desde qué RUL el sensor sale del rango basal
        for rul_threshold in range(200, 0, -1):
            mask = (train_df["rul_raw"] <= rul_threshold) & (train_df["rul_raw"] > rul_threshold - 10)
            if mask.sum() == 0:
                continue
            seg_mean = train_df.loc[mask, s].mean()
            if seg_mean > threshold_above or seg_mean < threshold_below:
                print(f"    {s:<6}: degradación observable a RUL ≈ {rul_threshold} ciclos")
                break

    # Varianza intra vs inter motor
    print("\n[3] Varianza intra-motor vs inter-motor (ANOVA one-way):")
    print("    (Ratio > 1 indica que la varianza entre motores explica")
    print("     más que la varianza dentro de cada motor — señal de degradación)")
    for s in ['s11','s4','s3','s12']:
        groups     = [g[s].values for _, g in train_df.groupby("engine_id")]
        grand_mean = train_df[s].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_within  = sum(((g - g.mean())**2).sum() for g in groups)
        ratio      = ss_between / (ss_within + 1e-10)
        print(f"    {s:<6}: SS_between/SS_within = {ratio:.4f}")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL — FASE 3
# ─────────────────────────────────────────────

def run_fase3():
    print("\n" + "═"*65)
    print("  FASE 3 — ANÁLISIS EXPLORATORIO ORIENTADO A DEGRADACIÓN")
    print("  NASA C-MAPSS | FD001 (+ comparativa FD003)")
    print("═"*65)

    # ── 1. Cargar datos ──
    print("\n[1/6] Cargando datos raw de FD001...")
    train_df = load_raw("FD001")
    print(f"      {train_df.shape[0]:,} filas | {train_df['engine_id'].nunique()} motores")

    # ── 2. Spaghetti plots ──
    print("\n[2/6] Generando spaghetti plots (20 motores × 8 sensores)...")
    plot_spaghetti(train_df,
        save_path=f"{FIG_DIR}/f3_01_spaghetti_plots.png")

    # ── 3. Heatmap de correlación ──
    print("\n[3/6] Generando heatmap de correlación (Pearson + Spearman)...")
    plot_correlation_heatmap(train_df,
        save_path=f"{FIG_DIR}/f3_02_correlation_heatmap.png")

    # ── 4. RUL por cuartil de vida ──
    print("\n[4/6] Análisis del RUL por cuartil de vida útil...")
    plot_rul_by_life_quartile(train_df,
        save_path=f"{FIG_DIR}/f3_03_rul_by_life_quartile.png")

    # ── 5. Curvas de degradación media ──
    print("\n[5/6] Generando curvas de degradación media por intervalo de RUL...")
    plot_degradation_phases(train_df,
        save_path=f"{FIG_DIR}/f3_04_degradation_phases.png")

    # ── 6. Comparativa FD001 vs FD003 ──
    # Necesita train_FD003.txt — lo copiamos si está disponible
    fd003_staged = "/mnt/user-data/uploads/PROYECT_04/archive/train_FD003.txt"
    fd003_dest   = f"{DATA_DIR}/train_FD003.txt"
    if os.path.exists(fd003_staged) and not os.path.exists(fd003_dest):
        import shutil
        shutil.copy(fd003_staged, fd003_dest)

    print("\n[6/6] Generando comparativa FD001 vs FD003...")
    plot_fd001_vs_fd003(
        save_path=f"{FIG_DIR}/f3_05_fd001_vs_fd003.png")

    # ── Análisis estadístico textual ──
    print_eda_summary(train_df)

    # ── Resumen ──
    life = train_df.groupby("engine_id")["cycle"].max()
    print("\n" + "═"*65)
    print("  RESUMEN FASE 3")
    print("═"*65)

    # Top sensores por correlación
    corrs = [(s, abs(pearsonr(train_df[s], train_df["rul_raw"])[0]))
             for s in INFORMATIVE]
    corrs.sort(key=lambda x: -x[1])
    print("  Top-5 sensores por |r| Pearson con RUL:")
    for s, r in corrs[:5]:
        print(f"    {s:<6} |r| = {r:.4f}  — {SENSOR_LABELS[s]}")

    print(f"\n  Sensores con señal ALTA (|r| > 0.6) : "
          f"{sum(1 for _,r in corrs if r>0.6)}")
    print(f"  Sensores con señal MEDIA (|r| > 0.3): "
          f"{sum(1 for _,r in corrs if 0.3<r<=0.6)}")
    print(f"  Figuras generadas : 5")
    print("═"*65)
    print("  ✅ FASE 3 COMPLETADA")
    print("═"*65 + "\n")


if __name__ == "__main__":
    run_fase3()
