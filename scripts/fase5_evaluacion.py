"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 5: Evaluación e interpretabilidad (sobre Fase 2 + Fase 4)
=============================================================================
Esta versión sustituye a la Fase 5 original, que RECONSTRUÍA desde cero
—dentro del propio script de evaluación— todo el pipeline de datos
(carga cruda + GlobalMinMaxScaler + rolling features), duplicando por
segunda vez (la primera fue la Fase 2 original) la misma lógica de
feature engineering. Aquí no se reconstruye nada: se cargan
directamente los parquets ya construidos por fase2_feature_engineering.py
(`train/test_FD001_features_112.parquet`, que ya contienen el RUL
verdadero en TODOS los ciclos de test, no solo el último) y los
artefactos ya entrenados por fase4_modelado.py (`xgboost_fd001.pkl`,
`final_scaler.pkl`).

Contenido:
  f5_01 — SHAP summary plot (beeswarm) + bar chart top-20 features
  f5_02 — SHAP dependence plots: top 3 features
  f5_03 — Curvas de degradación predicha vs. real (10 motores)
  f5_04 — Error absoluto por tercil de RUL (early/mid/late)
  f5_05 — Scatter predicho vs. real con densidad + bandas ±15 ciclos

CORRECCIÓN (sept. 2026): el modelo evaluado ya no está fijado a
"xgboost". Esa versión previa era un default heredado de una revisión
en la que XGBoost SÍ era el mejor modelo en test — pero esta auditoría
encontró (ver fase0_preprocesado.py / README, sección "discrepancia con
los números originales") que en este entorno **RandomForest** supera a
XGBoost en test RMSE (11.04 vs. 11.32), y Fase 5 seguía interpretando
el modelo equivocado sin que nada en el código lo verificara. Ahora
`get_best_model_name()` lee `fase4_resultados.csv` (generado por Fase
4 en esta misma ejecución) y determina el modelo con menor test RMSE
dinámicamente — así esta fase nunca puede quedar desincronizada de
cuál es realmente el mejor modelo, aunque cambie en una ejecución
futura (p. ej. por una versión distinta de una librería).

Referencia:
  - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to
    interpreting model predictions. NeurIPS 30. — SHAP / valores de
    Shapley como atribución aditiva de features.
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
np.random.seed(42)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fase0_preprocesado as f0
import fase2_feature_engineering as f2
import fase4_modelado as f4

OUTPUT_DIR = f0.OUTPUT_DIR
MODEL_DIR  = f0.MODEL_DIR
FIG_DIR    = f0.OUTPUT_DIR  # f5-f7 guardan sus figuras en outputs/, no en figures/ (convención heredada del proyecto original)
RUL_CAP    = f4.RUL_CAP

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
}
plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────
# 1. CARGA DE MODELO, SCALER Y DATOS (nada se reconstruye)
# ─────────────────────────────────────────────

def get_best_model_name(metric: str = "test_rmse") -> str:
    """
    Determina el mejor modelo de regresión leyendo
    outputs/fase4_resultados.csv (generado por fase4_modelado.py) —
    nunca hardcodeado, para que Fase 5 interprete siempre el modelo
    que realmente ganó en ESTA ejecución, no el que ganaba en una
    revisión anterior del proyecto (ver nota de cabecera del módulo).
    """
    results_path = f"{OUTPUT_DIR}/fase4_resultados.csv"
    if not os.path.exists(results_path):
        warnings.warn(
            f"No se encuentra {results_path} — no se puede determinar "
            "el mejor modelo automáticamente. Ejecuta fase4_modelado.py "
            "primero. Usando 'xgboost' como fallback."
        )
        return "xgboost"
    df = pd.read_csv(results_path)
    best_row = df.loc[df[metric].idxmin()]
    return str(best_row["modelo"]).lower()


def load_model_and_scaler(model_name: str = "xgboost"):
    model  = joblib.load(f"{MODEL_DIR}/{model_name}_fd001.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")
    return model, scaler


def load_last_cycle_data():
    """Igual que f4.load_features_112, pero devuelve también los
    DataFrames sin convertir a numpy (para poder anotar engine_id en
    los gráficos)."""
    X_train, y_train, groups, X_test, y_test, feat_cols = f4.load_features_112()
    return X_train, y_train, groups, X_test, y_test, feat_cols


def load_full_test_trajectories(feat_cols: list):
    """
    Carga TODOS los ciclos de test (no solo el último) desde el mismo
    parquet que ya generó Fase 2 — que ya trae el RUL verdadero
    calculado en toda la trayectoria por cuenta atrás
    (`compute_rul_test_full`). Los primeros ciclos de cada motor tienen
    NaN en las features de pendiente (ventana incompleta); se
    rellenan con 0 solo para poder graficar la curva completa, igual
    que hacía la Fase 5 original.
    """
    test_full = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_features_112.parquet")
    test_full = test_full.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return test_full


# ─────────────────────────────────────────────
# 2. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_fase5(model_name: str = None):
    if model_name is None:
        model_name = get_best_model_name()
        auto_detected = True
    else:
        auto_detected = False

    print("\n" + "═" * 70)
    print(f"  FASE 5 — EVALUACIÓN E INTERPRETABILIDAD ({model_name}"
          f"{' — auto-detectado como mejor modelo de Fase 4' if auto_detected else ''})")
    print("═" * 70)

    print("\n[1/6] Cargando modelo, scaler y datos (Fase 2 + Fase 4)...")
    model, scaler = load_model_and_scaler(model_name)
    X_train, y_train, groups, X_test, y_test, feat_cols = load_last_cycle_data()
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    y_pred = np.clip(model.predict(X_test_s), 0, RUL_CAP)
    rmse_test = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"      RMSE test ({model_name}) = {rmse_test:.3f} ciclos")

    print("\n[2/6] SHAP (interventional) — importancia de features...")
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train_s), size=min(500, len(X_train_s)), replace=False)
    explainer = shap.TreeExplainer(model, data=X_train_s[bg_idx],
                                   feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_s)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
    top20 = feat_importance.head(20)
    print(f"      Top 5: {list(top20.index[:5])}")

    plot_shap_summary(shap_values, X_test_s, feat_cols, top20,
                       save_path=f"{FIG_DIR}/f5_01_shap_summary.png")
    plot_shap_dependence(shap_values, X_test_s, feat_cols, list(top20.index[:3]), y_test,
                         save_path=f"{FIG_DIR}/f5_02_shap_dependence.png")

    print("\n[3/6] Curvas de degradación predicha vs. real (10 motores)...")
    test_full = load_full_test_trajectories(feat_cols)
    plot_degradation_curves(model, scaler, test_full, feat_cols,
                            save_path=f"{FIG_DIR}/f5_03_degradacion_predicha.png")

    print("\n[4/6] Error por tercil de RUL...")
    stats_tercil = plot_error_by_tercile(y_test, y_pred, groups=None,
                                        save_path=f"{FIG_DIR}/f5_04_error_por_tercil.png")
    print(stats_tercil.to_string())

    print("\n[5/6] Scatter predicho vs. real...")
    plot_scatter_density(y_test, y_pred, save_path=f"{FIG_DIR}/f5_05_scatter_predicho_real.png")

    print("\n[6/6] Exportando resumen (para que Fase 6 no reconstruya nada)...")
    top20.rename("mean_abs_shap").to_csv(f"{OUTPUT_DIR}/fase5_shap_top20.csv", header=True)
    stats_tercil.to_csv(f"{OUTPUT_DIR}/fase5_error_tercil.csv")
    pd.DataFrame([{"model_name": model_name, "rmse_test": rmse_test}]).to_csv(
        f"{OUTPUT_DIR}/fase5_resumen.csv", index=False)
    print(f"      fase5_shap_top20.csv, fase5_error_tercil.csv, fase5_resumen.csv")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 5")
    print("═" * 70)
    print(f"  Modelo evaluado      : {model_name}")
    print(f"  RMSE test            : {rmse_test:.3f} ciclos")
    print(f"  Top feature (SHAP)   : {top20.index[0]} ({top20.iloc[0]:.3f})")
    print("═" * 70)
    print("  ✅ FASE 5 COMPLETADA")
    print("═" * 70 + "\n")

    return feat_importance, stats_tercil, rmse_test


# ─────────────────────────────────────────────
# 3. FIGURAS
# ─────────────────────────────────────────────

def plot_shap_summary(shap_values, X_test_s, feat_cols, top20, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("f5_01 — SHAP: Importancia de características (test set)",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    top20_idx = [feat_cols.index(f) for f in top20.index]
    shap_top = shap_values[:, top20_idx]
    feat_top = X_test_s[:, top20_idx]
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.coolwarm
    rng = np.random.RandomState(42)
    for i, feat_name in enumerate(top20.index):
        sv = shap_top[:, i]
        fv = feat_top[:, i]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
        jitter = rng.uniform(-0.2, 0.2, size=len(sv))
        y_pos = (len(top20) - 1 - i) + jitter
        ax.scatter(sv, y_pos, c=cmap(norm(fv_norm)), s=20, alpha=0.7, linewidths=0)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(list(reversed(top20.index)), fontsize=8)
    ax.set_xlabel("Valor SHAP (impacto en predicción RUL, ciclos)")
    ax.set_title("Beeswarm — top 20 features", fontsize=11)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Valor normalizado de feature\n(azul=bajo, rojo=alto)", fontsize=8)

    ax2 = axes[1]
    colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))
    bars = ax2.barh(range(len(top20)), top20.values[::-1], color=colors_bar[::-1])
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(list(reversed(top20.index)), fontsize=8)
    ax2.set_xlabel("Mean |SHAP value| (ciclos)")
    ax2.set_title("Importancia media absoluta — top 20", fontsize=11)
    for bar, val in zip(bars, top20.values[::-1]):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_values, X_test_s, feat_cols, top3, y_test, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("f5_02 — SHAP Dependence plots: top 3 features (test set)",
                 fontsize=13, fontweight="bold")
    for ax, feat_name in zip(axes, top3):
        fi = feat_cols.index(feat_name)
        sv = shap_values[:, fi]
        fv = X_test_s[:, fi]
        sc = ax.scatter(fv, sv, c=y_test, cmap="RdYlGn", s=40, vmin=0, vmax=RUL_CAP,
                        alpha=0.8, linewidths=0)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        z = np.polyfit(fv, sv, 1)
        x_line = np.linspace(fv.min(), fv.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="black", linewidth=1.5,
                linestyle="--", label=f"tendencia (m={z[0]:.2f})")
        ax.set_xlabel(f"{feat_name}\n(valor escalado)", fontsize=9)
        ax.set_ylabel("SHAP value (ciclos)", fontsize=9)
        ax.set_title(feat_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("RUL real (ciclos)", fontsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_degradation_curves(model, scaler, test_full, feat_cols, n_motors=10, save_path=None):
    rul_df = pd.read_csv(f"{f0.DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])
    rul_true_all = rul_df["RUL_true"].values.clip(0, RUL_CAP)
    rul_sorted = np.argsort(rul_true_all)
    selected_idx = [rul_sorted[i] for i in [5, 15, 30, 45, 55, 65, 75, 85, 92, 98]]
    selected_motors = [i + 1 for i in selected_idx]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("f5_03 — Degradación predicha vs. real: 10 motores representativos",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

    for plot_i, eng_id in enumerate(selected_motors):
        ax = fig.add_subplot(gs[plot_i // 5, plot_i % 5])
        eng_data = test_full[test_full["engine_id"] == eng_id].sort_values("cycle")
        Xeng = np.nan_to_num(eng_data[feat_cols].values.astype(np.float32), nan=0.0)
        Xeng_s = scaler.transform(Xeng)
        y_pred_eng = np.clip(model.predict(Xeng_s), 0, RUL_CAP)
        rul_real = eng_data["RUL"].values
        cycles = eng_data["cycle"].values

        ax.plot(cycles, rul_real, color="#2196F3", linewidth=1.8, label="RUL real")
        ax.plot(cycles, y_pred_eng, color="#FF5722", linewidth=1.8, linestyle="--", label="RUL pred.")
        ax.fill_between(cycles, np.maximum(0, y_pred_eng - 15), np.minimum(RUL_CAP, y_pred_eng + 15),
                        alpha=0.15, color="#FF5722", label="±15 ciclos")
        ax.axhline(30, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        rmse_eng = np.sqrt(np.mean((y_pred_eng - rul_real) ** 2))
        ax.set_title(f"Motor {eng_id}  (RUL_final={rul_real[-1]:.0f})\nRMSE={rmse_eng:.1f}", fontsize=9)
        ax.set_xlabel("Ciclo", fontsize=8); ax.set_ylabel("RUL (ciclos)", fontsize=8)
        ax.set_ylim(0, RUL_CAP + 5); ax.tick_params(labelsize=7)
        if plot_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_tercile(y_test, y_pred, groups=None, save_path=None):
    abs_err = np.abs(y_pred - y_test)
    err_df = pd.DataFrame({"RUL_true": y_test, "RUL_pred": y_pred, "AE": abs_err,
                           "error_sign": y_pred - y_test})
    q33, q67 = np.percentile(y_test, [33, 67])
    err_df["tercil"] = pd.cut(err_df["RUL_true"], bins=[-1, q33, q67, RUL_CAP + 1],
                               labels=[f"Late\n(RUL<={q33:.0f})", f"Mid\n({q33:.0f}<RUL<={q67:.0f})",
                                      f"Early\n(RUL>{q67:.0f})"])
    stats_tercil = err_df.groupby("tercil", observed=True).agg(
        n=("AE", "count"), MAE=("AE", "mean"),
        RMSE=("AE", lambda x: np.sqrt(np.mean(x**2))),
        pct_within15=("AE", lambda x: (x <= 15).mean() * 100),
        bias=("error_sign", "mean"),
    ).round(2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("f5_04 — Análisis de error por fase de vida (tercil de RUL real)",
                 fontsize=13, fontweight="bold")
    colors_t = ["#E53935", "#FB8C00", "#43A047"]
    data_by_tercil = [err_df[err_df["tercil"] == t]["AE"].values for t in err_df["tercil"].cat.categories]
    ax = axes[0]
    bp = ax.boxplot(data_by_tercil, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_t):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
    ax.set_ylabel("Error absoluto (ciclos)"); ax.set_title("Distribución de AE por tercil")

    ax2 = axes[1]
    x = np.arange(len(stats_tercil)); w = 0.35
    ax2.bar(x - w/2, stats_tercil["RMSE"], w, label="RMSE", color="#1565C0", alpha=0.8)
    ax2.bar(x + w/2, stats_tercil["MAE"], w, label="MAE", color="#0288D1", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
    ax2.set_ylabel("Error (ciclos)"); ax2.set_title("RMSE y MAE por tercil"); ax2.legend()

    ax3 = axes[2]
    bias_vals = stats_tercil["bias"].values
    bar_colors = ["#43A047" if b >= 0 else "#E53935" for b in bias_vals]
    ax3.bar(x - w/2, bias_vals, w, color=bar_colors, alpha=0.8)
    ax3_r = ax3.twinx()
    ax3_r.bar(x + w/2, stats_tercil["pct_within15"].values, w, color="#7B1FA2", alpha=0.5)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(err_df["tercil"].cat.categories, fontsize=9)
    ax3.set_ylabel("Sesgo medio (ciclos)"); ax3_r.set_ylabel("% |error|<=15", color="#7B1FA2")
    ax3.set_title("Sesgo y cobertura ±15 ciclos por tercil")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return stats_tercil


def plot_scatter_density(y_test, y_pred, save_path=None):
    xy = np.vstack([y_test, y_pred])
    kde = gaussian_kde(xy)(xy)
    order = kde.argsort()
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(y_test[order], y_pred[order], c=kde[order], cmap="viridis", s=60, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Densidad KDE")
    lims = [0, RUL_CAP]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], alpha=0.12, color="red", label="±15 ciclos")
    z = np.polyfit(y_test, y_pred, 1)
    x_line = np.linspace(0, RUL_CAP, 100)
    ax.plot(x_line, np.polyval(z, x_line), "b-", linewidth=1.5, label=f"Regresión (m={z[0]:.2f})")
    rmse_t = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae_t = np.mean(np.abs(y_pred - y_test))
    r2_t = 1 - np.sum((y_pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    pct15 = (np.abs(y_pred - y_test) <= 15).mean() * 100
    ax.text(0.05, 0.95, f"RMSE = {rmse_t:.2f}\nMAE = {mae_t:.2f}\nR² = {r2_t:.4f}\n%|err|<=15 = {pct15:.0f}%",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDBDBD", alpha=0.9))
    ax.set_xlabel("RUL real (ciclos)"); ax.set_ylabel("RUL predicho (ciclos)")
    ax.set_title("f5_05 — RUL predicho vs. real (test set)", fontsize=12, fontweight="bold")
    ax.set_xlim(-2, RUL_CAP + 5); ax.set_ylim(-2, RUL_CAP + 5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_fase5()
