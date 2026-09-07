"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 7: Mejoras estadísticas (sobre Fase 2 + Fase 4 + Fase 5)
=============================================================================
Sustituye a fase7_mejoras.py + fase7_m6.py originales, que reconstruían
—por TERCERA vez en el proyecto— el pipeline de datos completo dentro
del propio script de mejoras estadísticas. Aquí no se reconstruye nada:
se reutilizan las funciones de fase0/fase2/fase4/fase5.

Mejora 1: Comparación pareada de pérdidas XGBoost vs. LightGBM/RF
Mejora 2: Métricas segregadas por censura (RUL_true >= 125)
Mejora 3: PHM score asimétrica (Saxena et al., 2008)
Mejora 4: SHAP tree_path_dependent vs. interventional + ablación de s14
Mejora 5: Durbin-Watson y Breusch-Pagan (test correcto, con chequeo de
          pseudo-replicación) sobre trayectorias completas
Mejora 6: Ablación de ventanas temporales (LightGBM, GroupKFold k=5)

Correcciones de rigor estadístico respecto a la versión original:
  - M1 ya NO se presenta como un test de Diebold-Mariano. Harvey,
    Leybourne & Newbold (1997) diseñaron su test (y la corrección de
    muestra pequeña) para comparar la pérdida de pronósticos
    SECUENCIALES de una única serie temporal, donde el diferencial de
    pérdida puede estar autocorrelado (particularmente con horizontes
    h>1) y su varianza debe estimarse con un estimador robusto a esa
    autocorrelación (HAC). Aquí se comparan 100 motores DISTINTOS e
    INDEPENDIENTES — no hay eje temporal entre ellos, así que no hay
    autocorrelación que corregir y la maquinaria HAC de Diebold-Mariano
    no pinta nada. Lo que se calcula es, honestamente, una comparación
    pareada de pérdidas cuadráticas (t de Student pareado) con un
    factor de ajuste de varianza para muestra pequeña que TOMA PRESTADA
    la forma funcional de la corrección HLN, sin heredar su
    justificación de fondo (autocorrelación de horizonte h). Se
    mantiene el cálculo — es una comparación legítima — pero con la
    etiqueta correcta.
  - M5 ahora implementa el test de Breusch-Pagan (1979) genuino:
    regresión auxiliar de RESIDUOS AL CUADRADO (no |residuo|, que es
    el test de Glejser — lo que calculaba la versión original) sobre
    el RUL, con el estadístico LM = n·R² ~ χ²(k). Además, se añade un
    chequeo de pseudo-replicación: como Durbin-Watson ya demuestra que
    los residuos consecutivos dentro de un motor están fuertemente
    autocorrelados, el test pooled sobre ~13.000 ciclos trata como
    independientes observaciones que no lo son, inflando la
    significancia. Se recalcula el mismo test agregando a un valor por
    motor (n=100) para comparar cuánto cambia el p-valor.

Referencias:
  - Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the
    equality of prediction mean squared errors. International Journal
    of Forecasting, 13(2), 281-291.
  - Breusch, T. S., & Pagan, A. R. (1979). A simple test for
    heteroscedasticity and random coefficient variation. Econometrica,
    47(5), 1287-1294.
  - Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). PHM 2008.
  - Lundberg, S. M., & Lee, S. I. (2017). NeurIPS 30.
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from scipy import stats
from scipy.stats import chi2, f as fdist, pearsonr, ttest_rel
from sklearn.model_selection import GroupKFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
import fase5_evaluacion as f5

OUTPUT_DIR = f0.OUTPUT_DIR
MODEL_DIR  = f0.MODEL_DIR
FIG_DIR    = f0.OUTPUT_DIR  # f5-f7 guardan sus figuras en outputs/, no en figures/ (convención heredada del proyecto original)
RUL_CAP    = f4.RUL_CAP


# ─────────────────────────────────────────────
# CARGA COMÚN (reutilizada por todas las mejoras)
# ─────────────────────────────────────────────

def load_all_models_and_data():
    """Carga los 4 modelos de Fase 4 + el scaler, y el split de test
    (último ciclo/motor) de Fase 2. Nada se reentrena."""
    scaler = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")
    X_train, y_train, groups, X_test, y_test, feat_cols = f4.load_features_112()
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {name: joblib.load(f"{MODEL_DIR}/{name.lower()}_fd001.pkl")
              for name in ["Ridge", "RandomForest", "XGBoost", "LightGBM"]}
    preds = {name: np.clip(m.predict(X_test_s), 0, RUL_CAP) for name, m in models.items()}
    return models, preds, scaler, X_train_s, y_train, X_test_s, y_test, feat_cols


# ════════════════════════════════════════════════════════════════════
# MEJORA 1 — Comparación pareada de pérdidas (antes "Diebold-Mariano")
# ════════════════════════════════════════════════════════════════════

def paired_loss_comparison(y_true, pred_a, pred_b):
    """
    Compara el error cuadrático de dos modelos sobre las MISMAS
    unidades de test (diseño pareado: mismo motor, mismo RUL real,
    dos predicciones). Es un t de Student pareado sobre e_a² − e_b²,
    con un factor de ajuste de varianza de muestra pequeña que toma
    prestada la forma de Harvey-Leybourne-Newbold (1997) para n
    pequeño (n=100 aquí) — pero SIN el componente que en el test DM
    original corrige autocorrelación de horizonte h en una serie
    temporal, porque aquí no hay serie temporal entre motores.
    """
    e_a = (pred_a - y_true) ** 2
    e_b = (pred_b - y_true) ** 2
    d = e_a - e_b
    n = len(d)

    t_naive, p_naive = ttest_rel(e_a, e_b)
    hln_style_corr = np.sqrt((n - 1 + 1 / n) / n)  # forma HLN para h=1
    t_adj = t_naive / hln_style_corr
    p_adj = 2 * stats.t.sf(abs(t_adj), df=n - 1)

    return {"n": n, "d_mean": d.mean(), "d_std": d.std(),
            "t_naive": t_naive, "p_naive": p_naive,
            "t_adj": t_adj, "p_adj": p_adj}


def run_m1(preds, y_test):
    print("\n[M1] Comparación pareada de pérdidas (XGBoost vs. LightGBM/RF)...")
    r_lgb = paired_loss_comparison(y_test, preds["XGBoost"], preds["LightGBM"])
    r_rf  = paired_loss_comparison(y_test, preds["XGBoost"], preds["RandomForest"])
    print(f"  XGB vs LGB: t_adj={r_lgb['t_adj']:.3f}  p_adj={r_lgb['p_adj']:.4f}  "
          f"({'Sig.' if r_lgb['p_adj'] < 0.05 else 'No sig.'})")
    print(f"  XGB vs RF : t_adj={r_rf['t_adj']:.3f}  p_adj={r_rf['p_adj']:.4f}  "
          f"({'Sig.' if r_rf['p_adj'] < 0.05 else 'No sig.'})")

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["XGB vs LGB", "XGB vs RF"]
    t_vals = [r_lgb["t_adj"], r_rf["t_adj"]]
    p_vals = [r_lgb["p_adj"], r_rf["p_adj"]]
    colors = ["#E53935" if p < 0.05 else "#43A047" for p in p_vals]
    bars = ax.bar(labels, t_vals, color=colors, alpha=0.85)
    for bar, t, p in zip(bars, t_vals, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"t={t:.2f}\np={p:.4f}",
                ha="center", va="bottom" if t >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("t ajustado (estilo HLN, n=100)")
    ax.set_title("M1 — Comparación pareada de pérdidas cuadráticas\n"
                 "(no es un test de Diebold-Mariano: no hay serie temporal entre motores)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_01_comparacion_pareada.png", dpi=150, bbox_inches="tight")
    plt.close()
    return {"xgb_vs_lgb": r_lgb, "xgb_vs_rf": r_rf}


# ════════════════════════════════════════════════════════════════════
# MEJORA 2 — Segregación por censura derecha
# ════════════════════════════════════════════════════════════════════

def metricas(y_t, y_p):
    e = y_p - y_t
    ae = np.abs(e)
    return {"n": len(y_t), "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": ae.mean(),
            "R2": 1 - np.sum(e ** 2) / np.sum((y_t - y_t.mean()) ** 2),
            "Bias": e.mean(), "pct_15": (ae <= 15).mean() * 100}


def run_m2(preds, y_test, rul_true_raw):
    print("\n[M2] Segregación por censura (RUL_true >= 125)...")
    censurado = rul_true_raw >= RUL_CAP
    rows = []
    for name, yp in preds.items():
        for label, mask in [("Global", slice(None)), ("No censurado", ~censurado), ("Censurado", censurado)]:
            m = metricas(y_test[mask], yp[mask])
            rows.append({"Modelo": name, "Grupo": label, **m})
    df_m2 = pd.DataFrame(rows)
    print(df_m2[df_m2["Modelo"] == "XGBoost"][["Grupo", "n", "RMSE", "MAE", "Bias"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(preds):
        sub = df_m2[df_m2["Modelo"] == name]
        ax.bar(np.arange(3) + i * 0.2, sub["RMSE"], width=0.2, label=name)
    ax.set_xticks(np.arange(3) + 0.3)
    ax.set_xticklabels(["Global", "No censurado", "Censurado"])
    ax.set_ylabel("RMSE (ciclos)")
    ax.set_title(f"M2 — RMSE por grupo de censura (n_censurado={censurado.sum()})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_02_censura.png", dpi=150, bbox_inches="tight")
    plt.close()
    return df_m2


# ════════════════════════════════════════════════════════════════════
# MEJORA 3 — PHM score asimétrica
# ════════════════════════════════════════════════════════════════════

def phm_score(y_true, y_pred):
    """
    Saxena et al. (2008): d = predicho - real. Sobreestimar RUL (d>=0,
    el modelo cree que queda más vida de la real) se penaliza con
    divisor 10 (más agresivo); subestimar (d<0, error conservador) se
    penaliza con divisor 13 (más suave).
    """
    d = y_pred - y_true
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum()


def run_m3(preds, y_test):
    print("\n[M3] PHM score asimétrica...")
    scores = {name: phm_score(y_test, yp) for name, yp in preds.items()}
    rmses = {name: np.sqrt(np.mean((yp - y_test) ** 2)) for name, yp in preds.items()}
    for name, s in sorted(scores.items(), key=lambda x: x[1]):
        print(f"  {name:<15}: PHM={s:8.1f}  RMSE={rmses[name]:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(scores, key=lambda n: scores[n])
    ax.bar(order, [scores[n] for n in order], color=["#EF5350", "#FF7043", "#66BB6A", "#42A5F5"])
    ax.set_ylabel("PHM Score (suma, menor=mejor)")
    ax.set_title("M3 — PHM Score asimétrica (Saxena et al., 2008)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_03_phm_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    return scores


# ════════════════════════════════════════════════════════════════════
# MEJORA 4 — SHAP tree_path_dependent vs. interventional + ablación s14
# ════════════════════════════════════════════════════════════════════

def run_m4(models, X_train_s, X_test_s, feat_cols, test_rmse_ref):
    print("\n[M4] SHAP tree_path_dependent vs. interventional + ablación s14...")
    xgb_model = models["XGBoost"]

    explainer_tpd = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")
    shap_tpd = explainer_tpd.shap_values(X_test_s)
    mean_tpd = pd.Series(np.abs(shap_tpd).mean(axis=0), index=feat_cols)

    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train_s), min(500, len(X_train_s)), replace=False)
    explainer_int = shap.TreeExplainer(xgb_model, data=X_train_s[bg_idx], feature_perturbation="interventional")
    shap_int = explainer_int.shap_values(X_test_s)
    mean_int = pd.Series(np.abs(shap_int).mean(axis=0), index=feat_cols)

    print(f"  Top 3 tree_path_dependent: {list(mean_tpd.sort_values(ascending=False).index[:3])}")
    print(f"  Top 3 interventional     : {list(mean_int.sort_values(ascending=False).index[:3])}")

    # Ablación de s14: reconstruir features SIN s14 reutilizando fase2
    print("  Reconstruyendo features sin s14 (reutilizando fase2.build_canonical_112)...")
    train_norm = pd.read_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet")
    test_norm  = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_normalized.parquet")
    rul_df     = pd.read_parquet(f"{OUTPUT_DIR}/RUL_FD001.parquet")
    sensors_no14 = [s for s in f0.INFORMATIVE_EXPECTED if s != "s14"]

    train_rul_no14 = f2.compute_rul_train(train_norm)
    test_rul_no14  = f2.compute_rul_test_full(test_norm, rul_df)
    train_f_no14, cols_no14 = f2.build_canonical_112(train_rul_no14, sensors_no14)
    test_f_no14,  _         = f2.build_canonical_112(test_rul_no14, sensors_no14)

    train_clean = train_f_no14.dropna(subset=cols_no14)
    X_tr_no14 = train_clean[cols_no14].values.astype(np.float32)
    y_tr_no14 = train_clean["RUL"].values.astype(np.float32)
    grp_no14  = train_clean["engine_id"].values
    test_last_no14 = (test_f_no14.dropna(subset=cols_no14 + ["RUL"])
                      .sort_values("cycle").groupby("engine_id").tail(1))
    X_te_no14 = test_last_no14[cols_no14].values.astype(np.float32)
    y_te_no14 = test_last_no14["RUL"].clip(upper=RUL_CAP).values.astype(np.float32)

    from sklearn.preprocessing import StandardScaler

    def xgb_no14_factory():
        return xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                reg_lambda=1.0, tree_method="hist", device="cpu",
                                verbosity=0, random_state=42)

    # Bucle manual (no sklearn.cross_val_score): la wrapper sklearn de
    # XGBRegressor en este entorno (xgboost 2.1.3 + scikit-learn 1.6.1)
    # no implementa __sklearn_tags__, y cross_val_score falla al
    # intentar leerlo. fase4.cross_validate_models ya evita esto con un
    # bucle GroupKFold manual; se reutiliza el mismo patrón aquí.
    gkf = GroupKFold(n_splits=5)
    fold_rmse = []
    for tr_idx, val_idx in gkf.split(X_tr_no14, y_tr_no14, grp_no14):
        sc_fold = StandardScaler().fit(X_tr_no14[tr_idx])
        m = xgb_no14_factory()
        m.fit(sc_fold.transform(X_tr_no14[tr_idx]), y_tr_no14[tr_idx])
        p = np.clip(m.predict(sc_fold.transform(X_tr_no14[val_idx])), 0, RUL_CAP)
        fold_rmse.append(np.sqrt(np.mean((p - y_tr_no14[val_idx]) ** 2)))
    cv_rmse_no14 = np.array(fold_rmse)

    sc_no14 = StandardScaler().fit(X_tr_no14)
    xgb_no14 = xgb_no14_factory()
    xgb_no14.fit(sc_no14.transform(X_tr_no14), y_tr_no14)
    y_pred_no14 = np.clip(xgb_no14.predict(sc_no14.transform(X_te_no14)), 0, RUL_CAP)
    rmse_no14 = np.sqrt(np.mean((y_pred_no14 - y_te_no14) ** 2))

    print(f"  CV RMSE  sin s14 = {cv_rmse_no14.mean():.3f} ± {cv_rmse_no14.std():.3f}")
    print(f"  Test RMSE sin s14 = {rmse_no14:.3f}  (con s14 = {test_rmse_ref:.3f})")
    print(f"  Δ Test RMSE = {rmse_no14 - test_rmse_ref:+.3f} ciclos")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top15 = mean_int.sort_values(ascending=False).head(15)
    axes[0].barh(top15.index[::-1], top15.values[::-1], color="#0288D1", alpha=0.85)
    axes[0].set_title("Top 15 SHAP (interventional)")
    axes[1].bar(["Con s14", "Sin s14"], [test_rmse_ref, rmse_no14], color=["#1565C0", "#B71C1C"])
    axes[1].set_ylabel("Test RMSE (ciclos)")
    axes[1].set_title(f"Ablación s14 — Δ={rmse_no14 - test_rmse_ref:+.3f}")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_04_shap_ablacion.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"mean_tpd": mean_tpd, "mean_int": mean_int,
            "cv_rmse_no14": cv_rmse_no14.mean(), "test_rmse_no14": rmse_no14,
            "delta_test_rmse": rmse_no14 - test_rmse_ref}


# ════════════════════════════════════════════════════════════════════
# MEJORA 5 — Durbin-Watson y Breusch-Pagan (correcto) + pseudo-replicación
# ════════════════════════════════════════════════════════════════════

def dw_stat(residuals):
    d = np.diff(residuals)
    return np.sum(d ** 2) / np.sum(residuals ** 2)


def breusch_pagan_test(residuals: np.ndarray, exog: np.ndarray) -> dict:
    """
    Breusch & Pagan (1979): regresión auxiliar de residuos AL CUADRADO
    (no |residuo| — eso es el test de Glejser) sobre las variables
    exógenas. Bajo H0 (homocedasticidad), LM = n·R² ~ χ²(k). También se
    reporta el F-test equivalente (más común en la práctica, mismo
    R² pero referencia F en vez de χ²).
    """
    n = len(residuals)
    e2 = residuals ** 2
    exog = exog.reshape(-1, 1) if exog.ndim == 1 else exog
    X = np.column_stack([np.ones(n), exog])
    k = exog.shape[1]
    beta = np.linalg.lstsq(X, e2, rcond=None)[0]
    fitted = X @ beta
    ss_reg = np.sum((fitted - e2.mean()) ** 2)
    ss_tot = np.sum((e2 - e2.mean()) ** 2)
    r2 = ss_reg / ss_tot
    LM = n * r2
    p_lm = 1 - chi2.cdf(LM, df=k)
    F = (r2 / k) / ((1 - r2) / (n - k - 1))
    p_f = 1 - fdist.cdf(F, k, n - k - 1)
    return {"n": n, "r2": r2, "LM": LM, "p_lm": p_lm, "F": F, "p_f": p_f, "beta": beta}


def run_m5(models, scaler, feat_cols):
    print("\n[M5] Durbin-Watson y Breusch-Pagan (test correcto)...")
    xgb_model = models["XGBoost"]
    test_full = f5.load_full_test_trajectories(feat_cols)
    rul_df2 = pd.read_csv(f"{f0.DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])

    dw_results, all_res, all_rul = [], [], []
    for eid in range(1, 101):
        eng = test_full[test_full["engine_id"] == eid].sort_values("cycle")
        Xeng = np.nan_to_num(eng[feat_cols].values.astype(np.float32), nan=0.0)
        yp = np.clip(xgb_model.predict(scaler.transform(Xeng)), 0, RUL_CAP)
        res = yp - eng["RUL"].values
        dw_results.append({"engine_id": eid, "DW": dw_stat(res), "n_cycles": len(eng)})
        all_res.append(res)
        all_rul.append(eng["RUL"].values)

    dw_df = pd.DataFrame(dw_results)
    all_res_flat = np.concatenate(all_res)
    all_rul_flat = np.concatenate(all_rul)
    print(f"  DW medio por motor: {dw_df['DW'].mean():.3f} ± {dw_df['DW'].std():.3f}")
    print(f"  Motores con DW < 1.5: {(dw_df['DW'] < 1.5).sum()} / 100")

    bp_pooled = breusch_pagan_test(all_res_flat, all_rul_flat)
    print(f"  Breusch-Pagan (pooled, n={bp_pooled['n']}): "
          f"LM={bp_pooled['LM']:.2f} p_lm={bp_pooled['p_lm']:.6f}  "
          f"F={bp_pooled['F']:.2f} p_f={bp_pooled['p_f']:.6f}")

    # Chequeo de pseudo-replicación: agregar a 1 valor por motor (n=100)
    per_engine = pd.DataFrame({"res2_mean": [np.mean(r ** 2) for r in all_res],
                               "rul_mean": [np.mean(r) for r in all_rul]})
    # sqrt para volver a "unidades de residuo" comparable en la regresión BP
    bp_per_engine = breusch_pagan_test(np.sqrt(per_engine["res2_mean"].values),
                                       per_engine["rul_mean"].values)
    print(f"  Breusch-Pagan (por motor, n={bp_per_engine['n']}): "
          f"LM={bp_per_engine['LM']:.2f} p_lm={bp_per_engine['p_lm']:.6f}")
    print(f"  -> Si el p-valor pooled es mucho menor que el agregado por motor, "
          f"la significancia pooled está inflada por pseudo-replicación "
          f"(los {bp_pooled['n']} ciclos NO son observaciones independientes: "
          f"DW={dw_df['DW'].mean():.2f} ya demuestra autocorrelación fuerte).")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(dw_df["DW"], bins=20, color="#1976D2", alpha=0.8)
    axes[0].axvline(2.0, color="black", linestyle="--", label="DW=2 (sin autocorr.)")
    axes[0].axvline(dw_df["DW"].mean(), color="red", label=f"Media={dw_df['DW'].mean():.2f}")
    axes[0].set_title("Distribución Durbin-Watson por motor"); axes[0].legend(fontsize=8)

    axes[1].scatter(all_rul_flat[::20], all_res_flat[::20], s=3, alpha=0.3, color="#546E7A")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("RUL real"); axes[1].set_ylabel("Residuo")
    axes[1].set_title(f"Breusch-Pagan pooled: p={bp_pooled['p_lm']:.4f}\n"
                      f"por motor (n=100): p={bp_per_engine['p_lm']:.4f}")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_05_residuos.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"dw_mean": dw_df["DW"].mean(), "bp_pooled": bp_pooled, "bp_per_engine": bp_per_engine}


# ════════════════════════════════════════════════════════════════════
# MEJORA 6 — Ablación de ventanas temporales (LightGBM)
# ════════════════════════════════════════════════════════════════════

def build_for_windows(train_rul, test_rul, sensor_cols, windows):
    """Reutiliza los helpers genéricos de fase2 (parametrizados por
    ventana) en vez de reimplementar el rolling una tercera vez."""
    train_f = f2.add_rolling_features(train_rul, sensor_cols, windows, f2.CANONICAL_STATS)
    train_f = f2.add_cumulative_delta(train_f, sensor_cols)
    test_f  = f2.add_rolling_features(test_rul, sensor_cols, windows, f2.CANONICAL_STATS)
    test_f  = f2.add_cumulative_delta(test_f, sensor_cols)
    cols = f2.feature_cols_for(sensor_cols, windows, f2.CANONICAL_STATS)
    return train_f, test_f, cols


def run_m6(sensor_cols):
    print("\n[M6] Ablación de ventanas temporales (LightGBM, GroupKFold k=5)...")
    train_norm = pd.read_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet")
    test_norm  = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_normalized.parquet")
    rul_df     = pd.read_parquet(f"{OUTPUT_DIR}/RUL_FD001.parquet")
    train_rul = f2.compute_rul_train(train_norm)
    test_rul  = f2.compute_rul_test_full(test_norm, rul_df)

    window_combos = {"[5]": [5], "[15]": [15], "[30]": [30], "[45]": [45],
                     "[5,15]": [5, 15], "[15,30]*": [15, 30], "[15,45]": [15, 45],
                     "[30,45]": [30, 45], "[5,15,30]": [5, 15, 30], "[15,30,45]": [15, 30, 45]}

    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=5)
    results = []
    for label, windows in window_combos.items():
        train_f, test_f, cols = build_for_windows(train_rul, test_rul, sensor_cols, windows)
        train_clean = train_f.dropna(subset=cols)
        X = train_clean[cols].values.astype(np.float32)
        y = train_clean["RUL"].values.astype(np.float32)
        groups = train_clean["engine_id"].values
        test_last = (test_f.dropna(subset=cols + ["RUL"])
                    .sort_values("cycle").groupby("engine_id").tail(1))
        X_te = test_last[cols].values.astype(np.float32)
        y_te = test_last["RUL"].clip(upper=RUL_CAP).values.astype(np.float32)

        sc = StandardScaler().fit(X)
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                  verbose=-1, random_state=42, n_jobs=-1)
        cv_scores = -cross_val_score(model, sc.transform(X), y, groups=groups, cv=gkf,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1)
        model.fit(sc.transform(X), y)
        yp = np.clip(model.predict(sc.transform(X_te)), 0, RUL_CAP)
        test_rmse = np.sqrt(np.mean((yp - y_te) ** 2))
        results.append({"Ventanas": label, "n_feats": len(cols),
                        "CV_RMSE": cv_scores.mean(), "CV_std": cv_scores.std(),
                        "Test_RMSE": test_rmse})
        print(f"  {label:<12} n_feats={len(cols):<4} CV={cv_scores.mean():.2f} Test={test_rmse:.2f}")

    abl_df = pd.DataFrame(results).sort_values("CV_RMSE")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#B71C1C" if "*" in r else "#1976D2" for r in abl_df["Ventanas"]]
    ax.barh(abl_df["Ventanas"], abl_df["CV_RMSE"], color=colors, alpha=0.85)
    ax.set_xlabel("CV RMSE (ciclos)")
    ax.set_title("M6 — Ablación de ventanas (rojo = configuración del proyecto)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/f7_06_ablacion_ventanas.png", dpi=150, bbox_inches="tight")
    plt.close()

    abl_df.to_csv(f"{OUTPUT_DIR}/fase7_ablacion_ventanas.csv", index=False)
    return abl_df


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_fase7():
    print("\n" + "═" * 70)
    print("  FASE 7 — MEJORAS ESTADÍSTICAS (sobre Fase 2/4/5)")
    print("═" * 70)

    models, preds, scaler, X_train_s, y_train, X_test_s, y_test, feat_cols = load_all_models_and_data()
    rul_df = pd.read_csv(f"{f0.DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])
    rul_true_raw = rul_df["RUL_true"].values

    m1 = run_m1(preds, y_test)
    m2 = run_m2(preds, y_test, rul_true_raw)
    m3 = run_m3(preds, y_test)
    test_rmse_xgb = np.sqrt(np.mean((preds["XGBoost"] - y_test) ** 2))
    m4 = run_m4(models, X_train_s, X_test_s, feat_cols, test_rmse_xgb)
    m5 = run_m5(models, scaler, feat_cols)
    m6 = run_m6(f0.INFORMATIVE_EXPECTED)

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 7")
    print("═" * 70)
    print(f"  M1 (comparación pareada): XGB vs LGB p_adj={m1['xgb_vs_lgb']['p_adj']:.4f}")
    print(f"  M2 (censura): ver df_m2")
    print(f"  M3 (PHM): mejor = {min(m3, key=lambda n: m3[n])}")
    print(f"  M4 (s14): Δ Test RMSE = {m4['delta_test_rmse']:+.3f} ciclos")
    print(f"  M5 (DW/BP): DW={m5['dw_mean']:.3f}  BP pooled p={m5['bp_pooled']['p_lm']:.6f}  "
          f"BP por-motor p={m5['bp_per_engine']['p_lm']:.6f}")
    print(f"  M6 (ventanas): mejor CV = {m6.iloc[0]['Ventanas']}")
    print("═" * 70)
    print("  ✅ FASE 7 COMPLETADA")
    print("═" * 70 + "\n")

    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6}


if __name__ == "__main__":
    run_fase7()
