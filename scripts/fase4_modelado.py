"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 4: Modelado predictivo de RUL (sobre Fase 2 — conjunto canónico)
=============================================================================
Esta versión sustituye a la Fase 4 original, que cargaba
train/test_FD001_features.parquet de la Fase 2 antigua (238 columnas,
normalización per-motor con leakage — ver Fase 0). Esta consume
train/test_FD001_features_112.parquet, el conjunto CANÓNICO construido
en fase2_feature_engineering.py, que es feature a feature el mismo que
reconstruían ad hoc fase5_evaluacion.py y fase7_mejoras.py para poder
usar los modelos ya guardados.

Decisión de escalado (importante, y distinta de lo que documentaba la
Fase 4 original): se ajusta un único `StandardScaler` sobre las 112
features y se aplica a los 4 modelos por igual (no solo a Ridge). La
Fase 4 original solo escalaba para Ridge y dejaba sin escalar a los
modelos de árbol — pero `models/final_scaler.pkl`, que ya usaban
fase5_evaluacion.py y fase7_mejoras.py para cargar los modelos
guardados, es un StandardScaler sobre las 112 features aplicado a
TODOS los modelos. Un reescalado monótono por columna no cambia las
predicciones de un árbol de decisión (los splits comparan umbrales, y
el orden relativo de los valores no cambia), así que para
RandomForest/XGBoost/LightGBM esto es un no-op funcional — pero hay que
reproducirlo igual, porque si el modelo se entrenó con features
escaladas, alimentarlo con features sin escalar en inferencia sería
incorrecto (los splits del árbol están en "unidades escaladas").

Estrategia de validación: GroupKFold(k=5, grupos=engine_id) — cada
motor es una serie temporal completa; una partición aleatoria simple
mezclaría ciclos del mismo motor entre train y validación, produciendo
leakage y RMSE artificialmente bajo (Li et al., 2018).

EXTENSIÓN (sept. 2026) — CLASIFICACIÓN. El objetivo 2 del proyecto
("etiquetar si el motor está en zona de riesgo, RUL<=30 -> fallo
inminente") tenía la columna `early_failure` calculada desde Fase 2,
pero ningún script del pipeline auditado entrenaba realmente un
clasificador — un hueco real frente al alcance pedido. Se añade aquí,
sin tocar el track de regresión: mismos datos (112 features
canónicas), mismo protocolo (GroupKFold k=5 por motor, reentrenamiento
final en 100% train + evaluación en el test oficial de 100 motores),
mismos 4 algoritmos en su variante de clasificación (Logistic
Regression en vez de Ridge; RandomForest/XGBoost/LightGBM
Classifier). Las clases están desbalanceadas (~15% positivos, ver
Fase 2), así que los 4 modelos usan peso de clase balanceado
(`class_weight="balanced"` / `scale_pos_weight`) — y la métrica que se
prioriza es Recall, no Accuracy: en mantenimiento predictivo, un falso
negativo (fallo inminente no detectado) es mucho más costoso que un
falso positivo (revisión de más), así que el objetivo del proyecto
pide explícitamente priorizar Recall sobre Precision.

Referencias:
  - Heimes, F. O. (2008). RUL cap a 125 ciclos. PHM 2008.
  - Li, X., Ding, Q., & Sun, J.-Q. (2018). Reliability Engineering &
    System Safety, 172, 1-11. — validación agrupada por motor.
  - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
  - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting
    system. KDD 2016.
  - Ke, G. et al. (2017). LightGBM. NeurIPS 30.
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
    Statistical Learning (2nd ed.). Springer — regularización Ridge,
    validación cruzada.
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fase0_preprocesado as f0
import fase2_feature_engineering as f2

OUTPUT_DIR = f0.OUTPUT_DIR
MODEL_DIR  = f0.MODEL_DIR
FIG_DIR    = f0.FIG_DIR

MODEL_COLORS = {"Ridge": "#6B7280", "RandomForest": "#10B981",
                "XGBoost": "#F59E0B", "LightGBM": "#2563EB"}
CLASS_MODEL_COLORS = {"LogisticRegression": "#6B7280", "RandomForest": "#10B981",
                       "XGBoost": "#F59E0B", "LightGBM": "#2563EB"}

N_FOLDS = 5
RUL_CAP = 125
EARLY_FAILURE_THRESH = 30   # ciclos — consistente con Fase 2/3


# ─────────────────────────────────────────────
# 1. CARGA DEL CONJUNTO CANÓNICO (Fase 2)
# ─────────────────────────────────────────────

def load_features_112():
    """
    Carga train/test_FD001_features_112.parquet (Fase 2, conjunto
    canónico) y devuelve (X_train, y_train, groups, X_test, y_test,
    feat_cols) — firma ORIGINAL, sin tocar, porque fase5_evaluacion.py
    y fase7_mejoras.py ya la desempaquetan así. Delega en
    load_features_112_ext() (extensión de clasificación) y descarta los
    dos targets adicionales para mantener el contrato exacto.
    """
    (X_train, y_train, _y_train_clf, groups,
     X_test, y_test, _y_test_clf, feat_cols) = load_features_112_ext()
    return X_train, y_train, groups, X_test, y_test, feat_cols


def load_features_112_ext():
    """
    Igual que load_features_112, pero además devuelve el target de
    clasificación (`early_failure`, ya calculado en Fase 2 como
    RUL_raw <= 30) — mismos X_train/X_test/groups para ambos tracks,
    así que CV y test comparan modelos de regresión y clasificación
    sobre exactamente la misma partición de motores. Descarta filas con
    NaN en las features (los primeros ciclos de cada motor, donde la
    pendiente OLS rolling no tiene ventana completa — ver
    ols_slope_vectorized en fase2). El test se reduce al último ciclo
    observado de cada motor, que es el único punto con RUL verdadero
    conocido.
    """
    train_path = f"{OUTPUT_DIR}/train_FD001_features_112.parquet"
    test_path  = f"{OUTPUT_DIR}/test_FD001_features_112.parquet"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise RuntimeError(
            "No se encuentran los parquets de Fase 2 (conjunto 112). "
            "Ejecuta fase2_feature_engineering.py primero."
        )
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)

    feat_cols = f2.feature_cols_for(f0.INFORMATIVE_EXPECTED,
                                     f2.CANONICAL_WINDOWS, f2.CANONICAL_STATS)
    assert len(feat_cols) == 112, f"Se esperaban 112 features, hay {len(feat_cols)}"

    train_clean = train.dropna(subset=feat_cols)
    X_train = train_clean[feat_cols].values.astype(np.float32)
    y_train = train_clean["RUL"].values.astype(np.float32)
    y_train_clf = train_clean["early_failure"].values.astype(int)
    groups  = train_clean["engine_id"].values

    test_last = (test.dropna(subset=feat_cols + ["RUL"])
                 .sort_values("cycle").groupby("engine_id").tail(1)
                 .reset_index(drop=True))
    X_test = test_last[feat_cols].values.astype(np.float32)
    y_test = test_last["RUL"].clip(upper=RUL_CAP).values.astype(np.float32)
    y_test_clf = test_last["early_failure"].values.astype(int)

    print(f"  Train: {X_train.shape}  (de {len(train)} filas, "
          f"{len(train) - len(train_clean)} descartadas por NaN)")
    print(f"  Test (último ciclo/motor): {X_test.shape[0]} muestras")
    print(f"  Target train — μ={y_train.mean():.1f}, σ={y_train.std():.1f}")
    print(f"  Target clasificación train — early_failure=1: "
          f"{y_train_clf.sum()}/{len(y_train_clf)} ({y_train_clf.mean()*100:.1f}%)")
    return (X_train, y_train, y_train_clf, groups,
            X_test, y_test, y_test_clf, feat_cols)


# ─────────────────────────────────────────────
# 2. MODELOS
# ─────────────────────────────────────────────

def build_models():
    """
    Hiperparámetros documentados en el README original del proyecto —
    se mantienen sin cambios para que esta Fase 4 sea comparable con
    los resultados ya publicados:

    Ridge (α=10): regularización L2 de orden de magnitud estándar para
      features normalizadas (Hastie et al., 2009).
    RandomForest (n=150, max_depth=15, min_samples_leaf=4,
      max_features=0.33): compromiso velocidad/varianza para CV×5
      (Breiman, 2001).
    XGBoost (lr=0.05, n=300, max_depth=6, subsample/colsample=0.8):
      parámetros de referencia para datos tabulares (Chen & Guestrin,
      2016), con regularización estocástica.
    LightGBM (lr=0.05, n=300, num_leaves=63 ≈ 2^6-1, equivalente
      aproximado a max_depth=6 con crecimiento leaf-wise; Ke et al.,
      2017).
    """
    return {
        "Ridge": Ridge(alpha=10, fit_intercept=True),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=15, min_samples_leaf=4,
            max_features=0.33, n_jobs=-1, random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, tree_method="hist", device="cpu",
            verbosity=0, random_state=42,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_samples=20, verbose=-1,
            random_state=42,
        ),
    }


# ─────────────────────────────────────────────
# 3. VALIDACIÓN CRUZADA GROUPKFOLD
# ─────────────────────────────────────────────

def cross_validate_models(X, y, groups, models):
    """
    GroupKFold(k=5): los motores del fold de validación son
    completamente distintos de los de entrenamiento — replica el
    escenario real (predecir RUL de motores no vistos).

    El StandardScaler se ajusta SOLO sobre el fold de train en cada
    iteración (nunca sobre el fold de validación) y se aplica a los 4
    modelos por igual, para que la comparación entre modelos no se vea
    afectada por si alguno recibe features en una escala distinta.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {name: {"rmse": [], "mae": [], "r2": [], "preds": [], "trues": []} for name in models}

    print(f"\n  GroupKFold CV (k={N_FOLDS}) — {len(np.unique(groups))} motores")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

        fold_rmse = []
        for name, model in models.items():
            model.fit(X_tr_s, y_tr)
            preds = np.clip(model.predict(X_val_s), 0, RUL_CAP)
            results[name]["rmse"].append(np.sqrt(mean_squared_error(y_val, preds)))
            results[name]["mae"].append(mean_absolute_error(y_val, preds))
            results[name]["r2"].append(r2_score(y_val, preds))
            results[name]["preds"].extend(preds.tolist())
            results[name]["trues"].extend(y_val.tolist())
            fold_rmse.append(f"{results[name]['rmse'][-1]:.2f}")
        print(f"  Fold {fold+1}: " + "  ".join(f"{n}={v}" for n, v in zip(models, fold_rmse)))

    return results


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO FINAL + EVALUACIÓN EN TEST
# ─────────────────────────────────────────────

def train_final_models(X_train, y_train, X_test, y_test, models):
    """
    Reentrena cada modelo sobre el 100% de train (con el StandardScaler
    ajustado también sobre el 100% de train) y evalúa en el test
    oficial. Guarda el scaler y cada modelo en models/ — son los
    mismos artefactos que ya consumían fase5_evaluacion.py y
    fase7_mejoras.py.
    """
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, f"{MODEL_DIR}/final_scaler.pkl")

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    test_results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = np.clip(model.predict(X_test_s), 0, RUL_CAP)
        test_results[name] = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
            "preds": preds,
        }
        joblib.dump(model, f"{MODEL_DIR}/{name.lower()}_fd001.pkl")

    return test_results


# ─────────────────────────────────────────────
# 4b. MODELOS DE CLASIFICACIÓN (extensión — objetivo 2 del proyecto)
# ─────────────────────────────────────────────

def build_classification_models(scale_pos_weight: float = 1.0):
    """
    Variante de clasificación de los mismos 4 algoritmos que la
    regresión (Ridge -> LogisticRegression; RandomForest/XGBoost/
    LightGBM ya tienen clasificador nativo). Mismos hiperparámetros de
    estructura (profundidad, nº de árboles, subsample) que sus
    contrapartes de regresión, para que la comparación regresión vs.
    clasificación no se confunda con una diferencia de capacidad del
    modelo — solo cambia la función de pérdida.

    Todos incorporan peso de clase balanceado: con ~15% de positivos
    (`early_failure`), un modelo sin ponderar tiende a maximizar
    accuracy prediciendo casi siempre "normal", justo el error que más
    cuesta en mantenimiento predictivo (un fallo inminente no detectado
    frente a una revisión de más).
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_leaf=4,
            max_features=0.33, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, tree_method="hist", device="cpu",
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", verbosity=0, random_state=42,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_samples=20, class_weight="balanced",
            verbose=-1, random_state=42,
        ),
    }


def cross_validate_classifiers(X, y, groups, models):
    """
    Mismo protocolo que cross_validate_models (GroupKFold k=5 por
    motor, StandardScaler ajustado solo en el fold de train), pero con
    métricas de clasificación: F1, Recall, Precision y ROC-AUC (esta
    última no depende del umbral 0.5 — evalúa la separabilidad de las
    probabilidades predichas). Se guardan también las probabilidades
    predichas para poder dibujar curvas ROC/PR agregadas.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {name: {"f1": [], "recall": [], "precision": [], "roc_auc": [],
                       "probas": [], "trues": []} for name in models}

    print(f"\n  GroupKFold CV (k={N_FOLDS}) — {len(np.unique(groups))} motores")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

        fold_f1 = []
        for name, model in models.items():
            model.fit(X_tr_s, y_tr)
            proba = model.predict_proba(X_val_s)[:, 1]
            preds = (proba >= 0.5).astype(int)
            results[name]["f1"].append(f1_score(y_val, preds, zero_division=0))
            results[name]["recall"].append(recall_score(y_val, preds, zero_division=0))
            results[name]["precision"].append(precision_score(y_val, preds, zero_division=0))
            results[name]["roc_auc"].append(roc_auc_score(y_val, proba))
            results[name]["probas"].extend(proba.tolist())
            results[name]["trues"].extend(y_val.tolist())
            fold_f1.append(f"{results[name]['f1'][-1]:.2f}")
        print(f"  Fold {fold+1}: " + "  ".join(f"{n}={v}" for n, v in zip(models, fold_f1)))

    return results


def train_final_classifiers(X_train, y_train_clf, X_test, y_test_clf, models):
    """
    Reentrena cada clasificador sobre el 100% de train y evalúa en el
    mismo test oficial (100 motores, último ciclo) que la regresión.
    Reutiliza un StandardScaler ajustado sobre X_train — matemáticamente
    idéntico al final_scaler.pkl ya guardado por train_final_models
    (mismo X_train), así que no se duplica el artefacto en disco.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    test_results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train_clf)
        proba = model.predict_proba(X_test_s)[:, 1]
        preds = (proba >= 0.5).astype(int)
        test_results[name] = {
            "f1": f1_score(y_test_clf, preds, zero_division=0),
            "recall": recall_score(y_test_clf, preds, zero_division=0),
            "precision": precision_score(y_test_clf, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test_clf, proba),
            "preds": preds,
            "proba": proba,
        }
        joblib.dump(model, f"{MODEL_DIR}/{name.lower()}_classifier_fd001.pkl")

    return test_results


# ─────────────────────────────────────────────
# 5. COMPARACIÓN CON EL README PUBLICADO
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────

def plot_cv_comparison(cv_results, save_path=None):
    """f4_01 — RMSE/MAE/R² por fold (boxplot + media) para los 4 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = list(cv_results.keys())
    colors = [MODEL_COLORS[n] for n in names]
    for ax, metric, mlabel in zip(axes, ["rmse", "mae", "r2"],
                                  ["RMSE (ciclos)", "MAE (ciclos)", "R²"]):
        vals = [cv_results[n][metric] for n in names]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (v, col) in enumerate(zip(vals, colors)):
            ax.scatter(np.random.normal(i + 1, 0.05, len(v)), v, color=col,
                      s=40, zorder=5, alpha=0.85, edgecolors="white", linewidths=0.5)
            mu = np.mean(v)
            ax.plot(i + 1, mu, "D", color=col, markersize=8,
                   markeredgecolor="black", markeredgewidth=0.8, zorder=6)
            ax.text(i + 1, mu + (0.8 if metric != "r2" else 0.01), f"{mu:.2f}",
                   ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(mlabel); ax.set_title(f"{mlabel} — GroupKFold CV (k={N_FOLDS})", pad=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Comparación de modelos — Validación cruzada por motor (GroupKFold)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_pred_vs_actual(cv_results, save_path=None):
    """f4_02 — Predicho vs. real, todos los folds de CV agregados."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (name, res) in zip(axes.flatten(), cv_results.items()):
        preds, trues = np.array(res["preds"]), np.array(res["trues"])
        rmse, r2 = np.mean(res["rmse"]), np.mean(res["r2"])
        hb = ax.hexbin(trues, preds, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="N.º ciclos")
        lim = (0, RUL_CAP)
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Predicción perfecta")
        ax.fill_between(lim, [lim[0]-20, lim[1]-20], [lim[0]+20, lim[1]+20],
                        color="#EF4444", alpha=0.08, label="±20 ciclos")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("RUL real"); ax.set_ylabel("RUL predicho")
        ax.set_title(f"{name}\nRMSE={rmse:.2f} | R²={r2:.4f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)
    fig.suptitle("Predicción vs. RUL real — todos los folds CV agregados",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_residual_analysis(cv_results, save_path=None):
    """f4_03 — Distribución de residuos (RUL_real − RUL_pred) por modelo."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (name, res) in zip(axes.flatten(), cv_results.items()):
        preds, trues = np.array(res["preds"]), np.array(res["trues"])
        residuos = trues - preds
        ax.hist(residuos, bins=60, color=MODEL_COLORS[name], alpha=0.75,
               edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(0, color="black", linewidth=1.5, label="Residuo=0")
        ax.axvline(residuos.mean(), color="#EF4444", linewidth=1.5, linestyle="--",
                  label=f"Media={residuos.mean():.2f}")
        ax.set_xlabel("Residuo (RUL_real − RUL_pred)"); ax.set_ylabel("Densidad")
        ax.set_title(f"{name}\nμ={residuos.mean():.2f} | σ={residuos.std():.2f}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Distribución de residuos — todos los folds CV", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_test_evaluation(test_results, y_test, save_path=None):
    """f4_04 — RMSE/MAE/tabla/mejor modelo sobre el test oficial (100 motores)."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    names = list(test_results.keys())
    colors = [MODEL_COLORS[n] for n in names]

    ax0 = fig.add_subplot(gs[0, 0])
    rmse_vals = [test_results[n]["rmse"] for n in names]
    bars = ax0.bar(names, rmse_vals, color=colors, alpha=0.80, edgecolor="white")
    for bar, v in zip(bars, rmse_vals):
        ax0.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
    ax0.set_ylabel("RMSE (ciclos)"); ax0.set_title("RMSE — Test set oficial"); ax0.grid(alpha=0.35, axis="y")

    ax1 = fig.add_subplot(gs[0, 1])
    mae_vals = [test_results[n]["mae"] for n in names]
    bars = ax1.bar(names, mae_vals, color=colors, alpha=0.80, edgecolor="white")
    for bar, v in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.2f}", ha="center", fontweight="bold")
    ax1.set_ylabel("MAE (ciclos)"); ax1.set_title("MAE — Test set oficial"); ax1.grid(alpha=0.35, axis="y")

    ax2 = fig.add_subplot(gs[1, 0]); ax2.axis("off")
    table_data = [[n, f"{test_results[n]['rmse']:.3f}", f"{test_results[n]['mae']:.3f}",
                  f"{test_results[n]['r2']:.4f}"] for n in names]
    table = ax2.table(cellText=table_data, colLabels=["Modelo", "RMSE", "MAE", "R²"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.2, 2.2)
    ax2.set_title("Métricas — Test set oficial (100 motores)", fontweight="bold", pad=15)

    ax3 = fig.add_subplot(gs[1, 1])
    best_name = min(test_results, key=lambda n: test_results[n]["rmse"])
    best_preds = test_results[best_name]["preds"]
    order = np.argsort(y_test)
    ax3.scatter(range(len(y_test)), y_test[order], color="#6B7280", s=30, alpha=0.7, label="RUL real")
    ax3.scatter(range(len(y_test)), best_preds[order], color=MODEL_COLORS[best_name],
               s=30, alpha=0.7, label=f"{best_name} predicho")
    ax3.set_xlabel("Motor (ordenado por RUL real)"); ax3.set_ylabel("RUL (ciclos)")
    ax3.set_title(f"Mejor modelo: {best_name}\nRMSE={test_results[best_name]['rmse']:.2f}", fontsize=9)
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    fig.suptitle("Evaluación en Test Set Oficial — NASA C-MAPSS FD001 (100 motores)",
                fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# FIGURAS — CLASIFICACIÓN
# ─────────────────────────────────────────────

def plot_classification_cv_comparison(cv_results, save_path=None):
    """f4_05 — F1/Recall/ROC-AUC por fold (boxplot + media) para los 4 clasificadores."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = list(cv_results.keys())
    colors = [CLASS_MODEL_COLORS[n] for n in names]
    for ax, metric, mlabel in zip(axes, ["f1", "recall", "roc_auc"],
                                  ["F1-score", "Recall", "ROC-AUC"]):
        vals = [cv_results[n][metric] for n in names]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (v, col) in enumerate(zip(vals, colors)):
            ax.scatter(np.random.normal(i + 1, 0.05, len(v)), v, color=col,
                      s=40, zorder=5, alpha=0.85, edgecolors="white", linewidths=0.5)
            mu = np.mean(v)
            ax.plot(i + 1, mu, "D", color=col, markersize=8,
                   markeredgecolor="black", markeredgewidth=0.8, zorder=6)
            ax.text(i + 1, mu + 0.01, f"{mu:.3f}",
                   ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(mlabel); ax.set_title(f"{mlabel} — GroupKFold CV (k={N_FOLDS})", pad=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Clasificación early_failure (RUL≤30) — Validación cruzada por motor",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_classification_test_evaluation(test_results, y_test_clf, save_path=None):
    """
    f4_06 — Evaluación en test oficial: matrices de confusión (una por
    modelo), curvas ROC y Precision-Recall superpuestas, y tabla de
    métricas. La curva PR es más informativa que ROC bajo desbalance de
    clases (~15% positivos) — es la que mejor refleja el compromiso
    Recall/Precision que pide priorizar el objetivo del proyecto.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.55, wspace=0.4)
    names = list(test_results.keys())

    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        cm = confusion_matrix(y_test_clf, test_results[name]["preds"])
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Fallo inminente"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Fallo inminente"], fontsize=7)
        ax.set_xlabel("Predicho", fontsize=8); ax.set_ylabel("Real", fontsize=8)
        ax.set_title(f"{name}\nRecall={test_results[name]['recall']:.3f}", fontsize=9)

    ax_roc = fig.add_subplot(gs[1:, :2])
    for name in names:
        fpr, tpr, _ = roc_curve(y_test_clf, test_results[name]["proba"])
        ax_roc.plot(fpr, tpr, color=CLASS_MODEL_COLORS[name], linewidth=2,
                    label=f"{name} (AUC={test_results[name]['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Azar")
    ax_roc.set_xlabel("Tasa de falsos positivos"); ax_roc.set_ylabel("Recall (TPR)")
    ax_roc.set_title("Curva ROC — Test set oficial"); ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    ax_pr = fig.add_subplot(gs[1:, 2:])
    base_rate = y_test_clf.mean()
    for name in names:
        prec, rec, _ = precision_recall_curve(y_test_clf, test_results[name]["proba"])
        ax_pr.plot(rec, prec, color=CLASS_MODEL_COLORS[name], linewidth=2, label=name)
    ax_pr.axhline(base_rate, color="black", linestyle="--", linewidth=1, alpha=0.5,
                  label=f"Azar (tasa base={base_rate:.2f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curva Precision-Recall — Test set oficial"); ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.3)

    fig.suptitle("Clasificación early_failure — Evaluación en Test Set Oficial (100 motores)",
                fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


README_REFERENCE = {
    "Ridge":        {"cv_rmse": 15.62, "test_rmse": 14.78, "test_r2": 0.864},
    "RandomForest": {"cv_rmse": 13.80, "test_rmse": 13.69, "test_r2": 0.883},
    "XGBoost":      {"cv_rmse": 12.93, "test_rmse": 12.80, "test_r2": 0.898},
    "LightGBM":     {"cv_rmse": 12.90, "test_rmse": 13.51, "test_r2": 0.886},
}


def compare_with_readme(cv_results, test_results):
    rows = []
    for name in cv_results:
        cv_mean = np.mean(cv_results[name]["rmse"])
        ref = README_REFERENCE[name]
        rows.append({
            "modelo": name,
            "cv_rmse_este_run": round(cv_mean, 3),
            "cv_rmse_readme": ref["cv_rmse"],
            "test_rmse_este_run": round(test_results[name]["rmse"], 3),
            "test_rmse_readme": ref["test_rmse"],
            "test_r2_este_run": round(test_results[name]["r2"], 4),
            "test_r2_readme": ref["test_r2"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL — FASE 4
# ─────────────────────────────────────────────

def run_fase4():
    print("\n" + "═" * 70)
    print("  FASE 4 — MODELADO PREDICTIVO DE RUL (sobre Fase 2 canónico)")
    print("  Objetivo 1: regresión RUL  |  Objetivo 2: clasificación early_failure")
    print("═" * 70)

    print("\n[1/9] Cargando features canónicas (Fase 2)...")
    (X_train, y_train, y_train_clf, groups,
     X_test, y_test, y_test_clf, feat_cols) = load_features_112_ext()

    # ── Objetivo 1: REGRESIÓN ──────────────────────────────────────────
    print("\n[2/9] [Regresión] Validación cruzada GroupKFold (k=5)...")
    models_cv = build_models()
    cv_results = cross_validate_models(X_train, y_train, groups, models_cv)

    print(f"\n  {'Modelo':<15} {'CV RMSE μ±σ':>16} {'CV MAE μ±σ':>16} {'CV R² μ':>10}")
    for name, res in cv_results.items():
        print(f"  {name:<15} {np.mean(res['rmse']):.2f} ± {np.std(res['rmse']):.2f}      "
              f"{np.mean(res['mae']):.2f} ± {np.std(res['mae']):.2f}      "
              f"{np.mean(res['r2']):.4f}")

    print("\n[3/9] [Regresión] Entrenamiento final (100% train) + evaluación en test oficial...")
    models_final = build_models()
    test_results = train_final_models(X_train, y_train, X_test, y_test, models_final)
    print(f"\n  {'Modelo':<15} {'Test RMSE':>10} {'Test MAE':>10} {'Test R²':>10}")
    for name, r in test_results.items():
        print(f"  {name:<15} {r['rmse']:>10.3f} {r['mae']:>10.3f} {r['r2']:>10.4f}")

    print("\n[4/9] [Regresión] Comparando con los números publicados en el README...")
    comparison = compare_with_readme(cv_results, test_results)
    print(comparison.to_string(index=False))

    # ── Objetivo 2: CLASIFICACIÓN ──────────────────────────────────────
    scale_pos_weight = (y_train_clf == 0).sum() / max((y_train_clf == 1).sum(), 1)
    print(f"\n[5/9] [Clasificación] Validación cruzada GroupKFold (k=5) "
          f"— scale_pos_weight={scale_pos_weight:.2f}...")
    models_clf_cv = build_classification_models(scale_pos_weight)
    cv_results_clf = cross_validate_classifiers(X_train, y_train_clf, groups, models_clf_cv)

    print(f"\n  {'Modelo':<19} {'CV F1 μ±σ':>14} {'CV Recall μ±σ':>16} {'CV ROC-AUC μ':>14}")
    for name, res in cv_results_clf.items():
        print(f"  {name:<19} {np.mean(res['f1']):.3f} ± {np.std(res['f1']):.3f}    "
              f"{np.mean(res['recall']):.3f} ± {np.std(res['recall']):.3f}      "
              f"{np.mean(res['roc_auc']):.4f}")

    print("\n[6/9] [Clasificación] Entrenamiento final (100% train) + evaluación en test oficial...")
    models_clf_final = build_classification_models(scale_pos_weight)
    test_results_clf = train_final_classifiers(X_train, y_train_clf, X_test, y_test_clf,
                                                models_clf_final)
    print(f"\n  {'Modelo':<19} {'Test F1':>10} {'Test Recall':>12} {'Test Precision':>15} {'Test ROC-AUC':>13}")
    for name, r in test_results_clf.items():
        print(f"  {name:<19} {r['f1']:>10.3f} {r['recall']:>12.3f} "
              f"{r['precision']:>15.3f} {r['roc_auc']:>13.4f}")

    # ── Figuras ─────────────────────────────────────────────────────────
    print("\n[7/9] Generando figuras...")
    plot_cv_comparison(cv_results, save_path=f"{FIG_DIR}/f4_01_cv_rmse_comparison.png")
    plot_pred_vs_actual(cv_results, save_path=f"{FIG_DIR}/f4_02_predicted_vs_actual.png")
    plot_residual_analysis(cv_results, save_path=f"{FIG_DIR}/f4_03_residual_analysis.png")
    plot_test_evaluation(test_results, y_test, save_path=f"{FIG_DIR}/f4_04_test_set_evaluation.png")
    plot_classification_cv_comparison(cv_results_clf, save_path=f"{FIG_DIR}/f4_05_classification_cv_comparison.png")
    plot_classification_test_evaluation(test_results_clf, y_test_clf,
                                         save_path=f"{FIG_DIR}/f4_06_classification_test_evaluation.png")
    print(f"      6 figuras guardadas en {FIG_DIR}/ (4 regresión + 2 clasificación)")

    # ── Guardado de resultados ─────────────────────────────────────────
    print("\n[8/9] Guardando resultados de regresión...")
    rows = []
    for name, res in cv_results.items():
        tr = test_results[name]
        rows.append({
            "modelo": name,
            "cv_rmse_mean": np.mean(res["rmse"]), "cv_rmse_std": np.std(res["rmse"]),
            "cv_mae_mean": np.mean(res["mae"]), "cv_mae_std": np.std(res["mae"]),
            "cv_r2_mean": np.mean(res["r2"]),
            "test_rmse": tr["rmse"], "test_mae": tr["mae"], "test_r2": tr["r2"],
        })
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/fase4_resultados.csv", index=False)
    comparison.to_csv(f"{OUTPUT_DIR}/fase4_comparacion_readme.csv", index=False)

    print("\n[9/9] Guardando resultados de clasificación...")
    rows_clf = []
    for name, res in cv_results_clf.items():
        tr = test_results_clf[name]
        rows_clf.append({
            "modelo": name,
            "cv_f1_mean": np.mean(res["f1"]), "cv_f1_std": np.std(res["f1"]),
            "cv_recall_mean": np.mean(res["recall"]), "cv_recall_std": np.std(res["recall"]),
            "cv_precision_mean": np.mean(res["precision"]),
            "cv_roc_auc_mean": np.mean(res["roc_auc"]),
            "test_f1": tr["f1"], "test_recall": tr["recall"],
            "test_precision": tr["precision"], "test_roc_auc": tr["roc_auc"],
        })
    df_clf = pd.DataFrame(rows_clf)
    df_clf.to_csv(f"{OUTPUT_DIR}/fase4_clasificacion_resultados.csv", index=False)

    best = min(test_results, key=lambda n: test_results[n]["rmse"])
    best_clf = max(test_results_clf, key=lambda n: test_results_clf[n]["recall"])
    print("\n" + "═" * 70)
    print("  RESUMEN FASE 4")
    print("═" * 70)
    print(f"  [Regresión] Mejor modelo (test RMSE): {best}")
    print(f"    RMSE={test_results[best]['rmse']:.3f}  MAE={test_results[best]['mae']:.3f}  "
          f"R²={test_results[best]['r2']:.4f}")
    print(f"  [Clasificación] Mejor modelo (test Recall, prioridad de seguridad): {best_clf}")
    print(f"    Recall={test_results_clf[best_clf]['recall']:.3f}  "
          f"F1={test_results_clf[best_clf]['f1']:.3f}  "
          f"ROC-AUC={test_results_clf[best_clf]['roc_auc']:.4f}")
    print(f"  Modelos y scaler guardados en: {MODEL_DIR}/")
    print("═" * 70)
    print("  ✅ FASE 4 COMPLETADA")
    print("═" * 70 + "\n")

    return cv_results, test_results, comparison, cv_results_clf, test_results_clf


if __name__ == "__main__":
    run_fase4()
