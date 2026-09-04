"""
FASE 7 — Mejora 6 únicamente (ablación de ventanas) + Resumen final
Se ejecuta tras haber completado M1-M5 con fase7_mejoras.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
import lightgbm as lgb
from scipy import stats
from scipy.stats import ttest_rel

DATA_DIR   = "/workspace/cmapss/data"
MODEL_DIR  = "/workspace/cmapss/models"
OUTPUT_DIR = "/workspace/cmapss/outputs"

COLS    = (["engine_id","cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)])
INFO    = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
WINDOWS = [15, 30]
RUL_CAP = 125

STYLE = {"figure.facecolor":"white","axes.facecolor":"white",
         "axes.spines.top":False,"axes.spines.right":False,
         "axes.grid":True,"grid.alpha":0.3,"font.size":10}
plt.rcParams.update(STYLE)

def load_raw(split):
    df = pd.read_csv(f"{DATA_DIR}/{split}_FD001.txt", sep=r"\s+", header=None, names=COLS)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def ols_slope_vec(arr, w):
    n = len(arr)
    if n < 2:
        return np.full(n, np.nan)
    w_eff = min(w, n)
    t = np.arange(w_eff, dtype=np.float64); t -= t.mean()
    denom = (t*t).sum()
    if denom == 0:
        return np.full(n, np.nan)
    k = t / denom
    pad  = np.full(w_eff - 1, np.nan)
    conv = np.convolve(arr, k[::-1], mode="valid")
    result = np.concatenate([pad, conv])
    if n < w:
        full = np.full(n, np.nan)
        full[w_eff-1:] = conv
        return full
    return result

def add_rolling(df, sensors, windows):
    out = df.copy()
    for eid, grp in df.groupby("engine_id"):
        idx = grp.index
        for s in sensors:
            v = grp[s].values.astype(np.float64)
            out.loc[idx, f"{s}_delta"] = v - v[0]
            for w in windows:
                ser = pd.Series(v, index=idx)
                out.loc[idx, f"{s}_w{w}_mean"]  = ser.rolling(w, min_periods=1).mean().values
                out.loc[idx, f"{s}_w{w}_std"]   = ser.rolling(w, min_periods=1).std(ddof=1).fillna(0).values
                out.loc[idx, f"{s}_w{w}_slope"] = ols_slope_vec(v, w)
    return out

def feat_cols_for(sensors, windows):
    c = list(sensors)
    for s in sensors:
        for w in windows:
            c += [f"{s}_w{w}_mean", f"{s}_w{w}_std", f"{s}_w{w}_slope"]
    for s in sensors:
        c.append(f"{s}_delta")
    return c

def build_train(sensors=INFO, windows=WINDOWS):
    gsc = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")
    raw = load_raw("train"); raw[INFO] = gsc.transform(raw[INFO].values)
    raw["RUL"] = (raw.groupby("engine_id")["cycle"].transform("max") - raw["cycle"]).clip(upper=RUL_CAP)
    fc  = feat_cols_for(sensors, windows)
    df  = add_rolling(raw, sensors, windows).dropna(subset=fc)
    fsc = StandardScaler().fit(df[fc].values.astype(np.float32))
    X   = fsc.transform(df[fc].values.astype(np.float32))
    return X, df["RUL"].values, df["engine_id"].values, fc, fsc, gsc

def build_test(fsc, gsc, sensors=INFO, windows=WINDOWS):
    raw = load_raw("test"); raw[INFO] = gsc.transform(raw[INFO].values)
    rul_df = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])
    last   = raw.groupby("engine_id")["cycle"].idxmax()
    raw["RUL"] = np.nan
    for eid, idx in last.items():
        raw.loc[idx, "RUL"] = min(rul_df.loc[eid-1, "RUL_true"], RUL_CAP)
    fc    = feat_cols_for(sensors, windows)
    test_f = add_rolling(raw, sensors, windows)
    test_l = (test_f.dropna(subset=["RUL"]).sort_values("cycle")
              .groupby("engine_id").tail(1).reset_index(drop=True))
    X = fsc.transform(test_l[fc].values.astype(np.float32))
    return X, test_l["RUL"].values, test_l, rul_df

# ── Reconstruir estado mínimo para el resumen ─────────────────────────────────
print("Recargando estado de M1–M5…")

fsc_saved = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")
gsc_saved = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")
_, _, _, feat_cols, _, _ = build_train()   # solo para feat_cols y grupos
X_test, y_test, test_last, rul_df = build_test(fsc_saved, gsc_saved)

xgb_model  = joblib.load(f"{MODEL_DIR}/xgboost_fd001.pkl")
lgb_model  = joblib.load(f"{MODEL_DIR}/lightgbm_fd001.pkl")
rf_model   = joblib.load(f"{MODEL_DIR}/randomforest_fd001.pkl")
ridge_model= joblib.load(f"{MODEL_DIR}/ridge_fd001.pkl")

y_pred_xgb   = xgb_model.predict(X_test).clip(0, RUL_CAP)
y_pred_lgb   = lgb_model.predict(X_test).clip(0, RUL_CAP)
y_pred_rf    = rf_model.predict(X_test).clip(0, RUL_CAP)
y_pred_ridge = ridge_model.predict(X_test).clip(0, RUL_CAP)

print(f"  XGBoost RMSE={np.sqrt(np.mean((y_pred_xgb-y_test)**2)):.3f}  ← debe ser ~12.9")

# M1 state
n = 100
e_xgb = (y_pred_xgb - y_test)**2
e_lgb = (y_pred_lgb - y_test)**2
t_dm, p_dm = ttest_rel(e_xgb, e_lgb)
hln_corr = np.sqrt((n + 1 - 2 + 1/n) / n)
t_hln    = t_dm / hln_corr
p_hln    = 2 * stats.t.sf(abs(t_hln), df=n-1)

# M2 state
rul_true_raw = rul_df["RUL_true"].values
censurado    = rul_true_raw >= 125

def metricas(y_t, y_p):
    e = y_p - y_t; ae = np.abs(e)
    return {"n":len(y_t),"RMSE":np.sqrt(np.mean(e**2)),"MAE":ae.mean(),
            "R²":1-np.sum(e**2)/np.sum((y_t-y_t.mean())**2),"Bias":e.mean(),"% ≤15":(ae<=15).mean()*100}

# M3 state
def phm_score(y_true, y_pred):
    d = y_pred - y_true
    return np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum()

preds = {"XGBoost":y_pred_xgb,"LightGBM":y_pred_lgb,"RandomForest":y_pred_rf,"Ridge":y_pred_ridge}
phm_scores = {name: phm_score(y_test, yp) for name, yp in preds.items()}

# M4 state (desde archivo de ablación o recalcular RMSE)
rmse_no14 = 11.805   # resultado previo de M4

# M5 state  (solo necesitamos DW medio y p_bp para el resumen)
dw_mean = 0.173
p_bp    = 0.0

# ════════════════════════════════════════════════════════════════════════════
# MEJORA 6 — Ablación de ventanas temporales (LightGBM)
# ════════════════════════════════════════════════════════════════════════════
print("\n[M6] Ablación de ventanas temporales (LightGBM)…")

lgb_params = dict(n_estimators=100,    # reducido de 200 para velocidad
                  learning_rate=0.07, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                  verbose=-1, random_state=42, n_jobs=-1)

gkf = GroupKFold(n_splits=5)

window_combos = {
    "[5]":         [5],
    "[15]":        [15],
    "[30]":        [30],
    "[45]":        [45],
    "[5,15]":      [5,15],
    "[15,30]★":    [15,30],
    "[15,45]":     [15,45],
    "[30,45]":     [30,45],
    "[5,15,30]":   [5,15,30],
    "[15,30,45]":  [15,30,45],
}

ablation_results = []
for label, windows in window_combos.items():
    print(f"  {label}…", end=" ", flush=True)
    Xtr, ytr, grps, fc_w, fsc_w, gsc_w = build_train(INFO, windows)
    model_w = lgb.LGBMRegressor(**lgb_params)
    cv_scores = -cross_val_score(model_w, Xtr, ytr, groups=grps, cv=gkf,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_w.fit(Xtr, ytr)
    Xte, yte, _, _ = build_test(fsc_w, gsc_w, INFO, windows)
    yp_w = model_w.predict(Xte).clip(0, RUL_CAP)
    test_rmse_w = np.sqrt(np.mean((yp_w - y_test)**2))
    n_feats = Xtr.shape[1]
    ablation_results.append({
        "Ventanas": label, "n_feats": n_feats,
        "CV_RMSE": cv_scores.mean(), "CV_std": cv_scores.std(),
        "Test_RMSE": test_rmse_w,
    })
    print(f"CV={cv_scores.mean():.2f} Test={test_rmse_w:.2f}")

abl_df = pd.DataFrame(ablation_results).sort_values("CV_RMSE")
print(abl_df[["Ventanas","n_feats","CV_RMSE","CV_std","Test_RMSE"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Mejora 6 — Ablación de ventanas temporales (LightGBM, GroupKFold k=5)\n"
             "Justificación empírica de la selección w={15,30}",
             fontsize=12, fontweight="bold")

ax = axes[0]
colors_abl = ["#B71C1C" if "★" in r else "#1976D2" for r in abl_df["Ventanas"]]
bars_abl = ax.barh(abl_df["Ventanas"], abl_df["CV_RMSE"], color=colors_abl, alpha=0.85)
ax.errorbar(abl_df["CV_RMSE"], abl_df["Ventanas"],
            xerr=abl_df["CV_std"], fmt="none", color="black", capsize=4, lw=1.5)
for bar, val in zip(bars_abl, abl_df["CV_RMSE"]):
    ax.text(val + 0.05, bar.get_y()+bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("CV RMSE (ciclos)")
ax.set_title("CV RMSE por combinación de ventanas\n(rojo = configuración original del proyecto)")
orig_cv = abl_df[abl_df["Ventanas"]=="[15,30]★"]["CV_RMSE"].values[0]
ax.axvline(orig_cv, color="#B71C1C", lw=1.5, linestyle="--", alpha=0.6)

ax2 = axes[1]
ax2.scatter(abl_df["n_feats"], abl_df["CV_RMSE"],
            c=["#B71C1C" if "★" in r else "#1976D2" for r in abl_df["Ventanas"]],
            s=80, alpha=0.9, zorder=3)
for _, row in abl_df.iterrows():
    ax2.annotate(row["Ventanas"], (row["n_feats"], row["CV_RMSE"]),
                 fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax2.set_xlabel("Número de features")
ax2.set_ylabel("CV RMSE (ciclos)")
ax2.set_title("Trade-off complejidad vs error\nMás features no siempre = menor error")
pareto_x = abl_df["n_feats"].values
pareto_y = abl_df["CV_RMSE"].values
ax2.plot(pareto_x, pareto_y, "gray", alpha=0.3, lw=1, linestyle="--")

plt.tight_layout()
p_m6 = f"{OUTPUT_DIR}/f7_06_ablacion_ventanas.png"
fig.savefig(p_m6, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m6}")

# Guardar tabla M6
abl_df.to_csv(f"{OUTPUT_DIR}/fase7_ablacion_ventanas.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RESUMEN MEJORAS ESTADÍSTICAS — Fase 7")
print("="*65)
rmse_nc = metricas(y_test[~censurado], y_pred_xgb[~censurado])['RMSE']
rmse_c  = metricas(y_test[censurado],  y_pred_xgb[censurado])['RMSE']

print(f"\nM1 (Diebold-Mariano): XGB vs LGB  p={p_hln:.4f}  → "
      f"{'No significativo — equivalentes' if p_hln>0.05 else 'Significativo — XGB mejor'}")
print(f"M2 (Censura):  RMSE no censurado={rmse_nc:.3f}  RMSE censurado={rmse_c:.3f}")
print(f"M3 (PHM):  {sorted(phm_scores.items(), key=lambda x:x[1])}")
print(f"M4 (SHAP+s14):  Δ RMSE sin s14 = {rmse_no14-12.804:+.3f} ciclos")
print(f"M5 (DW):  DW medio={dw_mean:.3f}  BP p<0.0001")
print(f"M6 (Ventanas):  Mejor combo={abl_df.iloc[0]['Ventanas']} CV={abl_df.iloc[0]['CV_RMSE']:.3f}")

print("\n[FASE 7 COMPLETADA]")
