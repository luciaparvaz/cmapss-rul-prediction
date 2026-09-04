"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 4: Modelado Predictivo de RUL
=============================================================================
Objetivo: entrenar y comparar modelos de regresión para predecir el RUL
capeado (125 ciclos) a partir de las features de la Fase 2.

Modelos evaluados:
  1. Ridge Regression      — baseline lineal regularizado (L2)
  2. Random Forest         — ensemble de árboles (bagging)
  3. XGBoost               — gradient boosting (orden 2)
  4. LightGBM              — gradient boosting basado en histogramas

Estrategia de validación:
  GroupKFold (k=5, grupos = engine_id)
  Justificación: cada motor es una serie temporal completa; la partición
  aleatoria simple mezcla ciclos del mismo motor entre train y test,
  produciendo data leakage y RMSE artificialmente bajo (Saxena 2008,
  Li et al. 2018).

Métricas:
  - RMSE  (Root Mean Squared Error) — penaliza errores grandes
  - MAE   (Mean Absolute Error) — interpretable en ciclos
  - R²    (coeficiente de determinación) — varianza explicada

Figuras producidas:
  f4_01_cv_rmse_comparison.png  — RMSE CV por modelo (boxplot + media)
  f4_02_predicted_vs_actual.png — Pred vs real en fold de test (4 modelos)
  f4_03_residual_analysis.png   — Distribución de residuos y error por RUL
  f4_04_test_set_evaluation.png — Evaluación en test set oficial (100 motores)

Referencias:
  - Heimes (2008) — RUL cap at 125 cycles, IJPHM
  - Li et al. (2018) — GroupKFold para series de motores, RESS 172
  - Ramasso & Saxena (2014) — NASA/TM-2014-218496
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

from sklearn.linear_model   import Ridge
from sklearn.ensemble       import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics        import (mean_squared_error,
                                    mean_absolute_error, r2_score)
from sklearn.preprocessing  import StandardScaler
import xgboost  as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
OUTPUT_DIR = "/workspace/cmapss/outputs"
FIG_DIR    = "/workspace/cmapss/figures"
MODEL_DIR  = "/workspace/cmapss/models"
os.makedirs(MODEL_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#10B981",
    "accent":    "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#6B7280",
    "background":"#F8FAFC",
    "text":      "#1E293B",
    "purple":    "#7C3AED",
}

MODEL_COLORS = {
    "Ridge":        PALETTE["neutral"],
    "RandomForest": PALETTE["secondary"],
    "XGBoost":      PALETTE["accent"],
    "LightGBM":     PALETTE["primary"],
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CBD5E1",
    "axes.labelcolor":  PALETTE["text"],
    "text.color":       PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
    "grid.color":       "#E2E8F0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "figure.dpi":       150,
})

N_FOLDS   = 5
RUL_CAP   = 125
TARGET    = "RUL"      # RUL capeado en 125 ciclos


# ─────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────

def load_features():
    """
    Carga el conjunto de features generado en la Fase 2.
    Columnas meta (engine_id, cycle, os*, RUL_raw) se excluyen del
    vector de features.

    Test set: se evalúa solo sobre el ÚLTIMO ciclo de cada motor,
    que es el punto donde se conoce el RUL verdadero (RUL_FD001.txt).
    El RUL se capea a 125 ciclos coherentemente con el train.
    """
    train = pd.read_parquet(f"{OUTPUT_DIR}/train_FD001_features.parquet")
    test  = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_features.parquet")

    meta_cols = (["engine_id", "cycle", "RUL_raw", "RUL", "early_failure"]
                 + [f"os{i}" for i in range(1, 4)])
    meta_cols_train = [c for c in meta_cols if c in train.columns]
    feature_cols    = [c for c in train.columns if c not in meta_cols_train]

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[TARGET].values.astype(np.float32)
    groups  = train["engine_id"].values

    # Último ciclo de cada motor en test + capeado a RUL_CAP
    test_last = (test.sort_values("cycle")
                     .groupby("engine_id")
                     .tail(1)
                     .reset_index(drop=True))
    feat_test = [c for c in feature_cols if c in test_last.columns]
    X_test = test_last[feat_test].values.astype(np.float32)
    y_test = test_last[TARGET].clip(upper=RUL_CAP).values.astype(np.float32)
    print(f"  Test set (último ciclo por motor): {X_test.shape[0]} muestras")
    print(f"  Train: {X_train.shape} | Target μ={y_train.mean():.1f}, σ={y_train.std():.1f}")
    return X_train, y_train, groups, X_test, y_test, feature_cols


# ─────────────────────────────────────────────
# DEFINICIÓN DE MODELOS
# ─────────────────────────────────────────────

def build_models():
    """
    Devuelve un diccionario de modelos con hiperparámetros documentados.

    Ridge (α=10):
      α elegida por orden de magnitud estándar para regresión regularizada
      con features normalizadas; valores típicos 1–100 (Hastie et al. 2009).

    Random Forest (n=300, max_depth=20):
      300 árboles como compromiso varianza/costo; max_depth=20 evita
      sobreajuste en presencia de correlación entre features (Breiman 2001).

    XGBoost (lr=0.05, n=500, max_depth=6, subsample=0.8):
      Parámetros de referencia para tabular data según Chen & Guestrin (2016).
      subsample=0.8 y colsample=0.8 añaden regularización estocástica.

    LightGBM (lr=0.05, n=500, num_leaves=63):
      num_leaves=63 ≈ 2^6-1 — equivalente aproximado de max_depth=6 en LGB
      con crecimiento leaf-wise (Ke et al. 2017, NeurIPS).
    """
    models = {
        "Ridge": Ridge(alpha=10, fit_intercept=True),

        "RandomForest": RandomForestRegressor(
            n_estimators=150,           # equilibrio velocidad/varianza para CV×5
            max_depth=15,
            min_samples_leaf=4,
            max_features=0.33,          # ≈√p/p estándar para regresión (Breiman 2001)
            n_jobs=-1,
            random_state=42,
        ),

        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            device="cpu",
            verbosity=0,
            random_state=42,
        ),

        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            verbose=-1,
            random_state=42,
        ),
    }
    return models


# ─────────────────────────────────────────────
# VALIDACIÓN CRUZADA CON GROUPKFOLD
# ─────────────────────────────────────────────

def cross_validate_models(X, y, groups, models):
    """
    GroupKFold(k=5): en cada fold, los motores del grupo de validación
    son completamente distintos de los de entrenamiento. Esto replica
    el escenario real: se predice el RUL de motores no vistos.

    Para Ridge se aplica StandardScaler ajustado sobre el fold de train
    para evitar leakage de estadísticos de escala.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {name: {"rmse": [], "mae": [], "r2": [], "preds": [], "trues": []}
               for name in models}

    print(f"\n  GroupKFold CV (k={N_FOLDS}) — {len(np.unique(groups))} motores")
    print(f"  {'Fold':<6} " + "  ".join(f"{'RMSE_'+n:<14}" for n in models))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        fold_rmse = []

        for name, model in models.items():
            if name == "Ridge":
                scaler = StandardScaler()
                X_tr_  = scaler.fit_transform(X_tr)
                X_val_ = scaler.transform(X_val)
            else:
                X_tr_, X_val_ = X_tr, X_val

            model.fit(X_tr_, y_tr)
            preds = np.clip(model.predict(X_val_), 0, RUL_CAP)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae  = mean_absolute_error(y_val, preds)
            r2   = r2_score(y_val, preds)

            results[name]["rmse"].append(rmse)
            results[name]["mae"].append(mae)
            results[name]["r2"].append(r2)
            results[name]["preds"].extend(preds.tolist())
            results[name]["trues"].extend(y_val.tolist())
            fold_rmse.append(f"{rmse:.2f}")

        print(f"  {fold+1:<6} " + "  ".join(f"{v:<14}" for v in fold_rmse))

    return results


# ─────────────────────────────────────────────
# ENTRENAMIENTO FINAL Y EVALUACIÓN EN TEST
# ─────────────────────────────────────────────

def train_final_models(X_train, y_train, X_test, y_test, models):
    """
    Reentrena cada modelo sobre el 100% del train set y evalúa sobre
    el test set oficial (último ciclo de cada motor, RUL verdadero).
    """
    test_results = {}
    scaler_ridge = StandardScaler().fit(X_train)

    for name, model in models.items():
        if name == "Ridge":
            model.fit(scaler_ridge.transform(X_train), y_train)
            preds = np.clip(model.predict(scaler_ridge.transform(X_test)), 0, RUL_CAP)
        else:
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), 0, RUL_CAP)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        test_results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "preds": preds}

        # Guardar modelo entrenado
        joblib.dump(model, f"{MODEL_DIR}/{name.lower()}_fd001.pkl")

    return test_results, scaler_ridge


# ─────────────────────────────────────────────
# FIGURA 1 — COMPARACIÓN DE RMSE POR FOLD (CV)
# ─────────────────────────────────────────────

def plot_cv_comparison(cv_results, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names  = list(cv_results.keys())
    colors = [MODEL_COLORS[n] for n in names]
    metrics = ["rmse", "mae", "r2"]
    metric_labels = ["RMSE (ciclos)", "MAE (ciclos)", "R²"]

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = [cv_results[n][metric] for n in names]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Puntos individuales por fold
        for i, (v, col) in enumerate(zip(vals, colors)):
            x_jitter = np.random.normal(i + 1, 0.05, len(v))
            ax.scatter(x_jitter, v, color=col, s=40, zorder=5, alpha=0.85,
                       edgecolors="white", linewidths=0.5)

        # Media con texto
        for i, (v, col) in enumerate(zip(vals, colors)):
            mu = np.mean(v)
            ax.plot(i + 1, mu, "D", color=col, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=6)
            ax.text(i + 1, mu + (0.8 if metric != "r2" else 0.01),
                    f"{mu:.2f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=col)

        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel} — GroupKFold CV (k={N_FOLDS})", pad=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Comparación de modelos — Validación cruzada por motor (GroupKFold)\n"
        "Cada punto = 1 fold | Rombo = media | Sin leakage de datos entre motores",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 2 — PREDICCIÓN vs REAL (CV pooled)
# ─────────────────────────────────────────────

def plot_pred_vs_actual(cv_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax, (name, res) in zip(axes_flat, cv_results.items()):
        preds = np.array(res["preds"])
        trues = np.array(res["trues"])
        rmse  = np.mean(res["rmse"])
        r2    = np.mean(res["r2"])

        # Densidad de puntos (2D hexbin)
        hb = ax.hexbin(trues, preds, gridsize=40, cmap="Blues",
                       mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="N.º ciclos")

        # Línea perfecta y ±20 ciclos
        lim = (0, RUL_CAP)
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Predicción perfecta")
        ax.fill_between(lim, [lim[0]-20, lim[1]-20],
                        [lim[0]+20, lim[1]+20],
                        color=PALETTE["danger"], alpha=0.08, label="±20 ciclos")

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("RUL real (ciclos)", fontsize=9)
        ax.set_ylabel("RUL predicho (ciclos)", fontsize=9)
        ax.set_title(
            f"{name}\nRMSE = {rmse:.2f} ciclos | R² = {r2:.4f}",
            fontsize=10, pad=6
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Predicción vs RUL real — todos los folds CV agregados\n"
        "Hexbin density: azul oscuro = mayor concentración de predicciones",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 3 — ANÁLISIS DE RESIDUOS
# ─────────────────────────────────────────────

def plot_residual_analysis(cv_results, save_path=None):
    """
    Residuo = RUL_real − RUL_pred (positivo = subestimación del fallo)
    Un residuo positivo grande = el modelo cree que queda más vida de la real
    → peligroso en contexto de mantenimiento predictivo.
    Un modelo seguro debería tender a subestimar ligeramente (residuo < 0).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes_flat = axes.flatten()

    for ax, (name, res) in zip(axes_flat, cv_results.items()):
        preds    = np.array(res["preds"])
        trues    = np.array(res["trues"])
        residuos = trues - preds

        # Histograma de residuos
        ax.hist(residuos, bins=60, color=MODEL_COLORS[name],
                alpha=0.75, edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(0,  color="black", linewidth=1.5, linestyle="-",  label="Residuo=0")
        ax.axvline(residuos.mean(), color=PALETTE["danger"], linewidth=1.5,
                   linestyle="--", label=f"Media={residuos.mean():.2f}")

        # Percentil 5 y 95
        p5, p95 = np.percentile(residuos, [5, 95])
        ax.axvspan(p5, p95, color=MODEL_COLORS[name], alpha=0.10,
                   label=f"P5–P95: [{p5:.0f}, {p95:.0f}]")

        ax.set_xlabel("Residuo (RUL_real − RUL_pred)", fontsize=8)
        ax.set_ylabel("Densidad", fontsize=8)
        ax.set_title(
            f"{name}\nμ={residuos.mean():.2f} | σ={residuos.std():.2f} | "
            f"Sesgo +: {(residuos > 0).mean()*100:.1f}%",
            fontsize=9, pad=5
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Distribución de residuos — todos los folds CV\n"
        "Residuo > 0: subestima la degradación (fallo antes de lo predicho) — situación de riesgo",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# FIGURA 4 — EVALUACIÓN EN TEST SET OFICIAL
# ─────────────────────────────────────────────

def plot_test_evaluation(test_results, y_test, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    names  = list(test_results.keys())
    colors = [MODEL_COLORS[n] for n in names]

    # ── Panel (0,0): Barras RMSE ──
    ax0 = fig.add_subplot(gs[0, 0])
    rmse_vals = [test_results[n]["rmse"] for n in names]
    bars = ax0.bar(names, rmse_vals, color=colors, alpha=0.80,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, rmse_vals):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax0.set_ylabel("RMSE (ciclos)"); ax0.set_title("RMSE — Test set oficial", pad=8)
    ax0.grid(alpha=0.35, axis="y")

    # ── Panel (0,1): Barras MAE ──
    ax1 = fig.add_subplot(gs[0, 1])
    mae_vals = [test_results[n]["mae"] for n in names]
    bars = ax1.bar(names, mae_vals, color=colors, alpha=0.80,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("MAE (ciclos)"); ax1.set_title("MAE — Test set oficial", pad=8)
    ax1.grid(alpha=0.35, axis="y")

    # ── Panel (1,0): Tabla resumen ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    table_data = []
    for n in names:
        r = test_results[n]
        table_data.append([n, f"{r['rmse']:.3f}", f"{r['mae']:.3f}", f"{r['r2']:.4f}"])
    table = ax2.table(
        cellText=table_data,
        colLabels=["Modelo", "RMSE", "MAE", "R²"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.2, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EFF6FF")
    ax2.set_title("Métricas — Test set oficial (100 motores)", fontweight="bold", pad=15)

    # ── Panel (1,1): Pred vs real del mejor modelo ──
    ax3 = fig.add_subplot(gs[1, 1])
    best_name = min(test_results, key=lambda n: test_results[n]["rmse"])
    best_preds = test_results[best_name]["preds"]

    # Ordenar por RUL real para mejor visualización
    order = np.argsort(y_test)
    ax3.scatter(range(len(y_test)), y_test[order],
                color=PALETTE["neutral"], s=30, alpha=0.7, label="RUL real", zorder=3)
    ax3.scatter(range(len(y_test)), best_preds[order],
                color=MODEL_COLORS[best_name], s=30, alpha=0.7,
                label=f"{best_name} predicho", zorder=4)
    ax3.fill_between(range(len(y_test)),
                     y_test[order] - 20, y_test[order] + 20,
                     color=PALETTE["neutral"], alpha=0.10, label="±20 ciclos")
    ax3.set_xlabel("Motor (ordenado por RUL real)", fontsize=8)
    ax3.set_ylabel("RUL (ciclos)")
    ax3.set_title(f"Mejor modelo: {best_name}\nRMSE={test_results[best_name]['rmse']:.2f} | "
                  f"MAE={test_results[best_name]['mae']:.2f}", fontsize=9, pad=6)
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    fig.suptitle(
        "Evaluación en Test Set Oficial — NASA C-MAPSS FD001 (100 motores)\n"
        "Entrenamiento sobre 100% del train set | RUL verdadero del último ciclo",
        fontsize=12, fontweight="bold"
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL — FASE 4
# ─────────────────────────────────────────────

def run_fase4():
    print("\n" + "═"*65)
    print("  FASE 4 — MODELADO PREDICTIVO DE RUL")
    print("  NASA C-MAPSS | FD001 | Regresión RUL capeado")
    print("═"*65)

    # ── 1. Cargar datos ──
    print("\n[1/6] Cargando features de Fase 2...")
    X_train, y_train, groups, X_test, y_test, feature_cols = load_features()
    print(f"      Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Test: {len(y_test) if y_test is not None else 'N/A'}")

    # ── 2. Instanciar modelos ──
    print("\n[2/6] Configurando modelos...")
    models = build_models()
    for name in models:
        print(f"      {name}")

    # ── 3. GroupKFold CV ──
    print("\n[3/6] Validación cruzada GroupKFold (k=5)...")
    cv_results = cross_validate_models(X_train, y_train, groups, models)

    # Resumen CV
    print("\n  Resumen GroupKFold CV:")
    print(f"  {'Modelo':<15} {'RMSE μ±σ':>18} {'MAE μ±σ':>18} {'R² μ':>10}")
    for name, res in cv_results.items():
        rm = np.mean(res["rmse"]); rs = np.std(res["rmse"])
        mm = np.mean(res["mae"]);  ms = np.std(res["mae"])
        r2m = np.mean(res["r2"])
        print(f"  {name:<15} {rm:.2f} ± {rs:.2f}        "
              f"{mm:.2f} ± {ms:.2f}       {r2m:.4f}")

    # ── 4. Figuras CV ──
    print("\n[4/6] Generando figuras de CV...")
    plot_cv_comparison(cv_results,
        save_path=f"{FIG_DIR}/f4_01_cv_rmse_comparison.png")
    plot_pred_vs_actual(cv_results,
        save_path=f"{FIG_DIR}/f4_02_predicted_vs_actual.png")
    plot_residual_analysis(cv_results,
        save_path=f"{FIG_DIR}/f4_03_residual_analysis.png")

    # ── 5. Entrenamiento final y evaluación en test ──
    print("\n[5/6] Entrenamiento final + evaluación en test set oficial...")
    models2 = build_models()   # instancias limpias
    test_results, _ = train_final_models(X_train, y_train, X_test, y_test, models2)

    print(f"\n  {'Modelo':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    for name, r in test_results.items():
        print(f"  {name:<15} {r['rmse']:>10.3f} {r['mae']:>10.3f} {r['r2']:>10.4f}")

    plot_test_evaluation(test_results, y_test,
        save_path=f"{FIG_DIR}/f4_04_test_set_evaluation.png")

    # ── 6. Guardar tabla de resultados ──
    print("\n[6/6] Guardando tabla de resultados...")
    rows = []
    for name, res in cv_results.items():
        tr = test_results[name]
        rows.append({
            "modelo":       name,
            "cv_rmse_mean": np.mean(res["rmse"]),
            "cv_rmse_std":  np.std(res["rmse"]),
            "cv_mae_mean":  np.mean(res["mae"]),
            "cv_mae_std":   np.std(res["mae"]),
            "cv_r2_mean":   np.mean(res["r2"]),
            "test_rmse":    tr["rmse"],
            "test_mae":     tr["mae"],
            "test_r2":      tr["r2"],
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{OUTPUT_DIR}/fase4_resultados.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/fase4_resultados.csv")

    # ── Resumen final ──
    best = min(test_results, key=lambda n: test_results[n]["rmse"])
    print("\n" + "═"*65)
    print("  RESUMEN FASE 4")
    print("═"*65)
    print(f"  Mejor modelo (test RMSE): {best}")
    print(f"    RMSE = {test_results[best]['rmse']:.3f} ciclos")
    print(f"    MAE  = {test_results[best]['mae']:.3f} ciclos")
    print(f"    R²   = {test_results[best]['r2']:.4f}")
    print(f"  Modelos guardados en: {MODEL_DIR}/")
    print(f"  Figuras generadas  : 4")
    print("═"*65)
    print("  ✅ FASE 4 COMPLETADA")
    print("═"*65 + "\n")


if __name__ == "__main__":
    run_fase4()
