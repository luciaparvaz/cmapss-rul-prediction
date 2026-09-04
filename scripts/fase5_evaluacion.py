"""
FASE 5 — Evaluación e interpretabilidad
NASA C-MAPSS FD001 | Mantenimiento Predictivo

Contenido:
  f5_01 — SHAP summary plot (beeswarm) + bar chart top-20 features
  f5_02 — SHAP dependence plots: s14_w30_mean, s11_w30_mean, s4_w30_mean
  f5_03 — Curvas de degradación predicha vs real (10 motores representativos)
  f5_04 — Error absoluto por tercil de RUL (early/mid/late)
  f5_05 — Scatter predicho vs real con densidad + bandas de error ±15 ciclos
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import shap
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR    = "/workspace/cmapss/data"
MODEL_DIR   = "/workspace/cmapss/models"
OUTPUT_DIR  = "/workspace/cmapss/outputs"

COLS = (["engine_id", "cycle"] +
        [f"op{i}" for i in range(1, 4)] +
        [f"s{i}"  for i in range(1, 22)])

INFO    = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
WINDOWS = [15, 30]
RUL_CAP = 125

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
}
plt.rcParams.update(STYLE)


# ── 1. Pipeline de datos (idéntico al de Fase 4 corregido) ───────────────────
def load_raw(split="train"):
    df = pd.read_csv(f"{DATA_DIR}/{split}_FD001.txt",
                     sep=r"\s+", header=None, names=COLS)
    # Eliminar columna vacía por doble espacio al final si existe
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df

def ols_slope_vectorized(arr, w):
    """Pendiente OLS vectorizada por convolución numpy (sin scipy)."""
    t = np.arange(w, dtype=np.float64)
    t -= t.mean()
    denom = (t * t).sum()
    kernel = t / denom
    pad = np.full(w - 1, np.nan)
    result = np.convolve(arr, kernel[::-1], mode="valid")
    return np.concatenate([pad, result])

def add_rolling_features(df: pd.DataFrame, sensors: list, windows: list) -> pd.DataFrame:
    out = df.copy()
    for eng_id, grp in df.groupby("engine_id"):
        idx = grp.index
        for s in sensors:
            vals = grp[s].values.astype(np.float64)
            # delta acumulado
            out.loc[idx, f"{s}_delta"] = vals - vals[0]
            for w in windows:
                ser = pd.Series(vals, index=idx)
                out.loc[idx, f"{s}_w{w}_mean"] = ser.rolling(w, min_periods=1).mean().values
                out.loc[idx, f"{s}_w{w}_std"]  = ser.rolling(w, min_periods=1).std(ddof=1).fillna(0).values
                out.loc[idx, f"{s}_w{w}_slope"] = ols_slope_vectorized(vals, w)
    return out

def build_feature_cols():
    cols = list(INFO)
    for s in INFO:
        for w in WINDOWS:
            cols += [f"{s}_w{w}_mean", f"{s}_w{w}_std", f"{s}_w{w}_slope"]
    for s in INFO:
        cols.append(f"{s}_delta")
    return cols

def build_pipeline(split="train"):
    """Reconstruye el pipeline usando los scalers guardados de Fase 4."""
    # Cargar scalers guardados (ajustados sobre train completo en Fase 4)
    gscaler = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")  # MinMaxScaler por sensor
    fscaler = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")           # StandardScaler sobre features

    raw = load_raw(split)
    raw[INFO] = gscaler.transform(raw[INFO].values)

    feat_cols = build_feature_cols()

    if split == "train":
        raw["RUL_raw"] = (raw.groupby("engine_id")["cycle"]
                          .transform("max") - raw["cycle"])
        raw["RUL"] = raw["RUL_raw"].clip(upper=RUL_CAP)
        train_f = add_rolling_features(raw, INFO, WINDOWS).dropna(subset=feat_cols)
        X = fscaler.transform(train_f[feat_cols].values.astype(np.float32))
        y = train_f["RUL"].values
        groups = train_f["engine_id"].values
        return X, y, groups, feat_cols, fscaler, train_f

    # Test: añadir RUL verdadero desde archivo
    rul_df = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])
    last_idx = raw.groupby("engine_id")["cycle"].idxmax()
    raw["RUL"] = np.nan
    for eng_id, idx in last_idx.items():
        rul_val = rul_df.loc[eng_id - 1, "RUL_true"]
        raw.loc[idx, "RUL"] = min(rul_val, RUL_CAP)

    test_f = add_rolling_features(raw, INFO, WINDOWS)
    test_last = (test_f.dropna(subset=["RUL"])
                 .sort_values("cycle")
                 .groupby("engine_id").tail(1)
                 .reset_index(drop=True))

    X = fscaler.transform(test_last[feat_cols].values.astype(np.float32))
    y = test_last["RUL"].values
    return X, y, test_last, feat_cols, fscaler


# ── 2. Pipeline para TRAYECTORIAS COMPLETAS de test ──────────────────────────
def build_test_full_trajectories():
    """Devuelve test con todos los ciclos usando scalers de Fase 4."""
    gscaler = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")
    fscaler = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")

    test_raw = load_raw("test")
    test_raw[INFO] = gscaler.transform(test_raw[INFO].values)

    feat_cols = build_feature_cols()
    test_f = add_rolling_features(test_raw, INFO, WINDOWS)

    return test_f, feat_cols, fscaler, None


# ── Carga modelo ─────────────────────────────────────────────────────────────
print("Cargando modelo XGBoost…")
with open(f"{MODEL_DIR}/xgboost_fd001.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# ── Datos test (último ciclo) ────────────────────────────────────────────────
print("Reconstruyendo pipeline de test…")
X_test, y_test, test_last, feat_cols, fscaler = build_pipeline("test")
y_pred = xgb_model.predict(X_test).clip(0, RUL_CAP)

# ── Datos train para SHAP ────────────────────────────────────────────────────
X_train, y_train, groups, feat_cols, fscaler_tr, train_f = build_pipeline("train")

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  RMSE test = {np.sqrt(np.mean((y_pred - y_test)**2)):.3f} ciclos")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f5_01 — SHAP summary beeswarm + bar chart (top 20 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[f5_01] Calculando SHAP values…")

# Muestra aleatoria de train para background (máx 500 filas)
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
X_bg   = X_train[bg_idx]

explainer   = shap.TreeExplainer(xgb_model, data=X_bg, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)          # shape (100, 112)

# Importancias medias
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
top20 = feat_importance.head(20)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("f5_01 — SHAP: Importancia de características (XGBoost | FD001 test set)",
             fontsize=13, fontweight="bold", y=1.01)

# Panel izquierdo: beeswarm manual
ax = axes[0]
top20_idx = [feat_cols.index(f) for f in top20.index]
shap_top = shap_values[:, top20_idx]        # (100, 20)
feat_top = X_test[:, top20_idx]             # (100, 20)

norm = Normalize(vmin=0, vmax=1)
cmap = plt.cm.coolwarm

for i, feat_name in enumerate(top20.index):
    col_idx = list(top20.index).index(feat_name)
    sv  = shap_top[:, col_idx]
    fv  = feat_top[:, col_idx]
    # Normalizar feature para colormap
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
    # Jitter vertical
    jitter = rng.uniform(-0.2, 0.2, size=len(sv))
    y_pos  = (len(top20) - 1 - i) + jitter
    colors = cmap(norm(fv_norm))
    ax.scatter(sv, y_pos, c=colors, s=20, alpha=0.7, linewidths=0)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f for f in reversed(top20.index)], fontsize=8)
ax.set_xlabel("Valor SHAP (impacto en predicción RUL, ciclos)")
ax.set_title("Beeswarm — top 20 features", fontsize=11)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Valor normalizado de feature\n(azul=bajo, rojo=alto)", fontsize=8)

# Panel derecho: bar chart importancias medias
ax2 = axes[1]
colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))
bars = ax2.barh(range(len(top20)), top20.values[::-1], color=colors_bar[::-1])
ax2.set_yticks(range(len(top20)))
ax2.set_yticklabels(list(reversed(top20.index)), fontsize=8)
ax2.set_xlabel("Mean |SHAP value| (ciclos)")
ax2.set_title("Importancia media absoluta — top 20", fontsize=11)
for i, (bar, val) in enumerate(zip(bars, top20.values[::-1])):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=7)

plt.tight_layout()
path_f5_01 = f"{OUTPUT_DIR}/f5_01_shap_summary.png"
fig.savefig(path_f5_01, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f5_01}")
print(f"  Top 5 features: {list(top20.index[:5])}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f5_02 — SHAP dependence plots: 3 features más importantes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[f5_02] Dependence plots…")

top3_features = list(top20.index[:3])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("f5_02 — SHAP Dependence plots: top 3 features (XGBoost | FD001 test)",
             fontsize=13, fontweight="bold")

for ax, feat_name in zip(axes, top3_features):
    fi  = feat_cols.index(feat_name)
    sv  = shap_values[:, fi]
    fv  = X_test[:, fi]

    # Interacción con feature de mayor correlación de SHAP
    # Usamos y_test (RUL real) como color de interacción
    sc = ax.scatter(fv, sv, c=y_test, cmap="RdYlGn", s=40,
                    vmin=0, vmax=RUL_CAP, alpha=0.8, linewidths=0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Línea de tendencia
    z = np.polyfit(fv, sv, 1)
    x_line = np.linspace(fv.min(), fv.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="black", linewidth=1.5,
            linestyle="--", label=f"tendencia (m={z[0]:.2f})")

    ax.set_xlabel(f"{feat_name}\n(valor normalizado)", fontsize=9)
    ax.set_ylabel("SHAP value (ciclos)", fontsize=9)
    ax.set_title(feat_name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("RUL real (ciclos)", fontsize=7)

plt.tight_layout()
path_f5_02 = f"{OUTPUT_DIR}/f5_02_shap_dependence.png"
fig.savefig(path_f5_02, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f5_02}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f5_03 — Curvas de degradación predicha vs real (10 motores)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[f5_03] Curvas de degradación predicha vs real…")

test_f_full, feat_cols2, fscaler2, train_f2 = build_test_full_trajectories()
rul_df = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])

# Seleccionar 10 motores representativos: distribución variada de RUL final
rul_true_all = rul_df["RUL_true"].values.clip(0, RUL_CAP)
rul_sorted   = np.argsort(rul_true_all)
# 2 de RUL bajo, 2 medio-bajo, 2 medio-alto, 2 alto, 2 muy alto
selected_idx = [rul_sorted[i] for i in [5, 15, 30, 45, 55, 65, 75, 85, 92, 98]]
selected_motors = [i + 1 for i in selected_idx]  # engine_id 1-based

fig = plt.figure(figsize=(20, 14))
fig.suptitle("f5_03 — Degradación predicha vs real: 10 motores representativos\n"
             "(XGBoost | FD001 | GlobalMinMaxScaler)",
             fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

for plot_i, eng_id in enumerate(selected_motors):
    ax = fig.add_subplot(gs[plot_i // 5, plot_i % 5])

    # Trayectoria completa de este motor en test
    eng_data = test_f_full[test_f_full["engine_id"] == eng_id].copy()
    eng_data = eng_data.sort_values("cycle").reset_index(drop=True)

    # Feature matrix (con posibles NaN en slope por inicio)
    Xeng = eng_data[feat_cols2].values.astype(np.float32)
    # Rellenar NaN con 0 para primeras filas (min_periods=1 para mean/std, pero slope tiene NaN)
    Xeng = np.nan_to_num(Xeng, nan=0.0)
    Xeng = fscaler2.transform(Xeng)

    y_pred_eng = xgb_model.predict(Xeng).clip(0, RUL_CAP)
    n_cycles   = len(eng_data)

    # RUL real: decrece desde n_cycles + rul_true hasta rul_true
    rul_final  = rul_df.loc[eng_id - 1, "RUL_true"]
    rul_real   = np.arange(n_cycles + rul_final, rul_final, -1)[:n_cycles].clip(0, RUL_CAP)
    cycles     = eng_data["cycle"].values

    ax.plot(cycles, rul_real,   color="#2196F3", linewidth=1.8, label="RUL real")
    ax.plot(cycles, y_pred_eng, color="#FF5722", linewidth=1.8, linestyle="--", label="RUL pred.")
    ax.fill_between(cycles,
                    np.maximum(0, y_pred_eng - 15),
                    np.minimum(RUL_CAP, y_pred_eng + 15),
                    alpha=0.15, color="#FF5722", label="±15 ciclos")
    ax.axhline(30, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    rmse_eng = np.sqrt(np.mean((y_pred_eng - rul_real)**2))
    ax.set_title(f"Motor {eng_id}  (RUL_final={min(rul_final,RUL_CAP)})\nRMSE={rmse_eng:.1f}",
                 fontsize=9)
    ax.set_xlabel("Ciclo", fontsize=8)
    ax.set_ylabel("RUL (ciclos)", fontsize=8)
    ax.set_ylim(0, RUL_CAP + 5)
    ax.tick_params(labelsize=7)
    if plot_i == 0:
        ax.legend(fontsize=7, loc="upper right")

path_f5_03 = f"{OUTPUT_DIR}/f5_03_degradacion_predicha.png"
fig.savefig(path_f5_03, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f5_03}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f5_04 — Error absoluto por tercil de RUL (early/mid/late)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[f5_04] Análisis de error por tercil de RUL…")

abs_err = np.abs(y_pred - y_test)
err_df = pd.DataFrame({
    "engine_id":  test_last["engine_id"].values,
    "RUL_true":   y_test,
    "RUL_pred":   y_pred,
    "AE":         abs_err,
    "error_sign": y_pred - y_test,  # positivo = sobrestima RUL (optimista)
})

# Terciles de RUL real
q33, q67 = np.percentile(y_test, [33, 67])
err_df["tercil"] = pd.cut(err_df["RUL_true"],
                           bins=[-1, q33, q67, RUL_CAP + 1],
                           labels=["Late\n(RUL≤{:.0f})".format(q33),
                                   "Mid\n({:.0f}<RUL≤{:.0f})".format(q33, q67),
                                   "Early\n(RUL>{:.0f})".format(q67)])

stats_tercil = err_df.groupby("tercil", observed=True).agg(
    n=("AE", "count"),
    MAE=("AE", "mean"),
    RMSE=("AE", lambda x: np.sqrt(np.mean(x**2))),
    pct_within15=("AE", lambda x: (x <= 15).mean() * 100),
    bias=("error_sign", "mean"),
).round(2)
print(stats_tercil)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("f5_04 — Análisis de error por fase de vida (tercil de RUL real)\n"
             "XGBoost | FD001 test set (n=100 motores)",
             fontsize=13, fontweight="bold")

# Panel 1: Boxplot AE por tercil
ax = axes[0]
colors_t = ["#E53935", "#FB8C00", "#43A047"]
data_by_tercil = [err_df[err_df["tercil"] == t]["AE"].values for t in err_df["tercil"].cat.categories]
bp = ax.boxplot(data_by_tercil, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors_t):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
ax.set_ylabel("Error absoluto (ciclos)")
ax.set_title("Distribución de error absoluto por tercil")

# Añadir n y MAE como anotaciones
for i, (tercil, row) in enumerate(stats_tercil.iterrows()):
    ax.text(i + 1, ax.get_ylim()[1] * 0.95,
            f"n={row['n']}\nMAE={row['MAE']:.1f}",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# Panel 2: RMSE y MAE por tercil (barras dobles)
ax2 = axes[1]
x = np.arange(len(stats_tercil))
w = 0.35
bars1 = ax2.bar(x - w/2, stats_tercil["RMSE"], w, label="RMSE",
                color="#1565C0", alpha=0.8)
bars2 = ax2.bar(x + w/2, stats_tercil["MAE"],  w, label="MAE",
                color="#0288D1", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
ax2.set_ylabel("Error (ciclos)")
ax2.set_title("RMSE y MAE por tercil")
ax2.legend()
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", fontsize=8)

# Panel 3: Sesgo (bias) y % within 15 ciclos
ax3 = axes[2]
bias_vals = stats_tercil["bias"].values
pct15_vals = stats_tercil["pct_within15"].values
bar_colors = ["#43A047" if b >= 0 else "#E53935" for b in bias_vals]
bars3 = ax3.bar(x - w/2, bias_vals, w, color=bar_colors, alpha=0.8, label="Sesgo (pred−real)")
ax3_r = ax3.twinx()
ax3_r.bar(x + w/2, pct15_vals, w, color="#7B1FA2", alpha=0.5, label="% within ±15 ciclos")
ax3_r.set_ylabel("% motores con |error|≤15 ciclos", color="#7B1FA2")
ax3_r.tick_params(axis="y", labelcolor="#7B1FA2")
ax3_r.set_ylim(0, 110)

ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
ax3.set_ylabel("Sesgo medio (ciclos)")
ax3.set_title("Sesgo y cobertura ±15 ciclos por tercil")
for bar, val in zip(bars3, bias_vals):
    ax3.text(bar.get_x() + bar.get_width()/2,
             val + (0.5 if val >= 0 else -0.5),
             f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
for i, val in enumerate(pct15_vals):
    ax3_r.text(i + w/2, val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9,
               color="#7B1FA2")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

plt.tight_layout()
path_f5_04 = f"{OUTPUT_DIR}/f5_04_error_por_tercil.png"
fig.savefig(path_f5_04, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f5_04}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f5_05 — Scatter predicho vs real con densidad + líneas de referencia
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[f5_05] Scatter predicho vs real…")

# KDE-style scatter con coloring por densidad
from scipy.stats import gaussian_kde

xy   = np.vstack([y_test, y_pred])
kde  = gaussian_kde(xy)(xy)
order = kde.argsort()

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(y_test[order], y_pred[order], c=kde[order],
                cmap="viridis", s=60, alpha=0.85, linewidths=0)
plt.colorbar(sc, ax=ax, label="Densidad KDE")

# Línea perfecta
lims = [0, RUL_CAP]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
# Bandas ±15 ciclos
ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims],
                alpha=0.12, color="red", label="±15 ciclos")

# Regresión lineal
z = np.polyfit(y_test, y_pred, 1)
x_line = np.linspace(0, RUL_CAP, 100)
ax.plot(x_line, np.polyval(z, x_line), "b-", linewidth=1.5,
        label=f"Regresión (m={z[0]:.2f}, b={z[1]:.1f})")

# Métricas
rmse_t = np.sqrt(np.mean((y_pred - y_test)**2))
mae_t  = np.mean(abs_err)
r2_t   = 1 - np.sum((y_pred - y_test)**2) / np.sum((y_test - y_test.mean())**2)
pct15  = (abs_err <= 15).mean() * 100

ax.text(0.05, 0.95,
        f"RMSE = {rmse_t:.2f} ciclos\nMAE  = {mae_t:.2f} ciclos\nR²   = {r2_t:.4f}\n% |err|≤15 = {pct15:.0f}%",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDBDBD", alpha=0.9))

ax.set_xlabel("RUL real (ciclos)", fontsize=11)
ax.set_ylabel("RUL predicho (ciclos)", fontsize=11)
ax.set_title("f5_05 — RUL predicho vs real\nXGBoost | FD001 test set (n=100 motores)",
             fontsize=12, fontweight="bold")
ax.set_xlim(-2, RUL_CAP + 5)
ax.set_ylim(-2, RUL_CAP + 5)
ax.legend(fontsize=9)

plt.tight_layout()
path_f5_05 = f"{OUTPUT_DIR}/f5_05_scatter_predicho_real.png"
fig.savefig(path_f5_05, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f5_05}")


# ── Resumen Fase 5 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN FASE 5 — Evaluación e interpretabilidad")
print("="*60)
print(f"\nMétricas test (XGBoost, n=100 motores):")
print(f"  RMSE  = {rmse_t:.3f} ciclos")
print(f"  MAE   = {mae_t:.3f} ciclos")
print(f"  R²    = {r2_t:.4f}")
print(f"  %±15c = {pct15:.0f}%")

print(f"\nTop 5 features SHAP:")
for i, (feat, val) in enumerate(top20.head(5).items()):
    print(f"  {i+1}. {feat:30s} = {val:.3f}")

print(f"\nError por tercil de RUL:")
print(stats_tercil[["n","MAE","RMSE","pct_within15","bias"]].to_string())

print(f"\nFiguras generadas:")
for i, p in enumerate([path_f5_01, path_f5_02, path_f5_03, path_f5_04, path_f5_05], 1):
    print(f"  f5_0{i}: {p}")
print("\n[FASE 5 COMPLETADA]")
