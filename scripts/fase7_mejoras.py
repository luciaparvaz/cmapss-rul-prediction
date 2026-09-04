"""
FASE 7 — Mejoras estadísticas (Mejoras 1–6)
NASA C-MAPSS FD001 | Mantenimiento Predictivo

Mejora 1: Test Diebold-Mariano XGBoost vs LightGBM
Mejora 2: Métricas segregadas por censura (RUL_true ≥ 125)
Mejora 3: PHM score asimétrica (Saxena et al. 2008) + re-ranking
Mejora 4: SHAP tree_path_dependent + ablación s14
Mejora 5: Durbin-Watson y Breusch-Pagan sobre trayectorias completas
Mejora 6: Ablación de ventanas temporales con LightGBM
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import pearsonr, ttest_rel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import shap

DATA_DIR   = "/home/claude/cmapss/data"
MODEL_DIR  = "/home/claude/cmapss/models"
OUTPUT_DIR = "/home/claude/cmapss/outputs"

COLS    = (["engine_id","cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)])
INFO    = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
WINDOWS = [15, 30]
RUL_CAP = 125

STYLE = {"figure.facecolor":"white","axes.facecolor":"white",
         "axes.spines.top":False,"axes.spines.right":False,
         "axes.grid":True,"grid.alpha":0.3,"font.size":10}
plt.rcParams.update(STYLE)

# ── Pipeline base (reutilizado de Fase 4/5) ──────────────────────────────────
def load_raw(split):
    df = pd.read_csv(f"{DATA_DIR}/{split}_FD001.txt", sep=r"\s+", header=None, names=COLS)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def ols_slope_vec(arr, w):
    """OLS slope vectorizado; maneja arrays más cortos que la ventana."""
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
    # Si el array es más corto que w original, rellenar con NaN al inicio
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

# ── Cargar modelos guardados ──────────────────────────────────────────────────
print("Cargando modelos y pipeline…")
# build_train: para feat_cols, grupos y datos de entrenamiento (M4/M6 usan scaler fresco)
X_train, y_train, groups, feat_cols, fsc_fresh, gsc_fresh = build_train()

# Para predecir con modelos GUARDADOS (M1/M2/M3) se cargan los scalers exactos
# con los que fueron entrenados en Fase 4 — usar uno nuevo produce RMSE ~19.8 (bug)
fsc_saved = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")
gsc_saved = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")
X_test,  y_test,  test_last, rul_df = build_test(fsc_saved, gsc_saved)

xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_fd001.pkl")
lgb_model  = joblib.load(f"{MODEL_DIR}/lightgbm_fd001.pkl")
rf_model   = joblib.load(f"{MODEL_DIR}/randomforest_fd001.pkl")
ridge_model= joblib.load(f"{MODEL_DIR}/ridge_fd001.pkl")

y_pred_xgb   = xgb_model.predict(X_test).clip(0, RUL_CAP)
y_pred_lgb   = lgb_model.predict(X_test).clip(0, RUL_CAP)
y_pred_rf    = rf_model.predict(X_test).clip(0, RUL_CAP)
y_pred_ridge = ridge_model.predict(X_test).clip(0, RUL_CAP)

print(f"  XGBoost  RMSE={np.sqrt(np.mean((y_pred_xgb -y_test)**2)):.3f}")
print(f"  LightGBM RMSE={np.sqrt(np.mean((y_pred_lgb -y_test)**2)):.3f}")

# ════════════════════════════════════════════════════════════════════════════
# MEJORA 1 — Test Diebold-Mariano XGBoost vs LightGBM
# ════════════════════════════════════════════════════════════════════════════
print("\n[M1] Diebold-Mariano…")

e_xgb = (y_pred_xgb - y_test)**2
e_lgb = (y_pred_lgb - y_test)**2
d     = e_xgb - e_lgb          # diferencial de pérdida

# Test DM: t-test sobre el diferencial (Harvey et al. 1997)
t_dm, p_dm = ttest_rel(e_xgb, e_lgb)
# Corrección de Harvey-Leybourne-Newbold (HLN) para muestras pequeñas
n = len(d)
hln_corr = np.sqrt((n + 1 - 2 + 1/n) / n)
t_hln    = t_dm / hln_corr
p_hln    = 2 * stats.t.sf(abs(t_hln), df=n-1)

print(f"  DM  t={t_dm:.3f}  p={p_dm:.4f}")
print(f"  HLN t={t_hln:.3f}  p={p_hln:.4f}")
print(f"  {'NO hay diferencia significativa' if p_hln>0.05 else 'Diferencia significativa'} (α=0.05)")

# También RF vs XGBoost
e_rf = (y_pred_rf - y_test)**2
t_rf, p_rf = ttest_rel(e_xgb, e_rf)
hln_rf = np.sqrt((n + 1 - 2 + 1/n) / n)
t_hln_rf = t_rf / hln_rf
p_hln_rf = 2 * stats.t.sf(abs(t_hln_rf), df=n-1)
print(f"  XGB vs RF: HLN t={t_hln_rf:.3f}  p={p_hln_rf:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mejora 1 — Test Diebold-Mariano (HLN): comparación estadística de modelos",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.scatter(range(n), d, color=["#EF5350" if v>0 else "#66BB6A" for v in d],
           s=30, alpha=0.7)
ax.axhline(0, color="black", lw=1)
ax.axhline(d.mean(), color="#1565C0", lw=1.5, linestyle="--",
           label=f"Media = {d.mean():.2f}")
ax.fill_between(range(n),
                d.mean() - 1.96*d.std()/np.sqrt(n),
                d.mean() + 1.96*d.std()/np.sqrt(n),
                alpha=0.15, color="#1565C0")
ax.set_xlabel("Motor (test)"); ax.set_ylabel("d = e²_XGB − e²_LGB (ciclos²)")
ax.set_title("Diferencial de pérdida cuadrática\npor motor (rojo = XGB peor)")
ax.legend()

# Tabla de resultados
ax2 = axes[1]; ax2.axis("off")
comparisons = [
    ["XGBoost vs LightGBM", f"{t_hln:.3f}", f"{p_hln:.4f}",
     "No sig." if p_hln > 0.05 else "Sig. (p<0.05)"],
    ["XGBoost vs RandomForest", f"{t_hln_rf:.3f}", f"{p_hln_rf:.4f}",
     "No sig." if p_hln_rf > 0.05 else "Sig. (p<0.05)"],
]
tab = ax2.table(
    cellText=comparisons,
    colLabels=["Comparación", "t (HLN)", "p-valor", "Conclusión"],
    cellLoc="center", loc="center",
    colWidths=[0.38, 0.17, 0.17, 0.28]
)
tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 2.8)
for j in range(4):
    tab[(0,j)].set_facecolor("#37474F"); tab[(0,j)].set_text_props(color="white", fontweight="bold")
# Colorear conclusión
for i in range(1, 3):
    concl = comparisons[i-1][3]
    color = "#E8F5E9" if "No sig" in concl else "#FFEBEE"
    tab[(i,3)].set_facecolor(color)
    tab[(i,3)].set_text_props(fontweight="bold")

# Nota interpretativa
interp = ("No sig." if p_hln > 0.05
          else "Significativa (α=0.05)")
ax2.text(0.5, 0.05,
         f"Test HLN (Harvey et al. 1997) — corrección para n=100\n"
         f"XGB vs LGB: diferencia {interp}\n"
         f"→ Ambos modelos son estadísticamente equivalentes" if p_hln > 0.05 else
         f"→ XGBoost es significativamente mejor",
         ha="center", va="bottom", transform=ax2.transAxes,
         fontsize=9, style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", alpha=0.8))
ax2.set_title("Resultados del test (α = 0.05)", fontsize=10, fontweight="bold")

plt.tight_layout()
p_m1 = f"{OUTPUT_DIR}/f7_01_diebold_mariano.png"
fig.savefig(p_m1, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m1}")


# ════════════════════════════════════════════════════════════════════════════
# MEJORA 2 — Métricas segregadas: censuradas vs no censuradas
# ════════════════════════════════════════════════════════════════════════════
print("\n[M2] Censura derecha…")

rul_true_raw = rul_df["RUL_true"].values  # sin cap
censurado    = rul_true_raw >= 125

def metricas(y_t, y_p):
    e  = y_p - y_t
    ae = np.abs(e)
    return {
        "n":      len(y_t),
        "RMSE":   np.sqrt(np.mean(e**2)),
        "MAE":    ae.mean(),
        "R²":     1 - np.sum(e**2)/np.sum((y_t - y_t.mean())**2),
        "Bias":   e.mean(),
        "% ≤15":  (ae<=15).mean()*100,
    }

preds = {"XGBoost": y_pred_xgb, "LightGBM": y_pred_lgb,
         "RandomForest": y_pred_rf, "Ridge": y_pred_ridge}

rows = []
for name, yp in preds.items():
    m_all  = metricas(y_test, yp)
    m_nc   = metricas(y_test[~censurado], yp[~censurado])
    m_c    = metricas(y_test[censurado],  yp[censurado])
    rows.append({"Modelo": name, "Grupo": f"Global (n={m_all['n']})", **m_all})
    rows.append({"Modelo": name, "Grupo": f"No censurado (n={m_nc['n']})", **m_nc})
    rows.append({"Modelo": name, "Grupo": f"Censurado RUL≥125 (n={m_c['n']})", **m_c})

df_m2 = pd.DataFrame(rows)
print(df_m2[df_m2["Modelo"]=="XGBoost"][["Grupo","RMSE","MAE","R²","Bias"]].to_string(index=False))

# XGBoost únicamente para la figura
xgb_rows = df_m2[df_m2["Modelo"]=="XGBoost"].copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Mejora 2 — Segregación por censura (RUL_true ≥ 125 = censurado)\n"
             "Todos los modelos | n_total=100, n_no_censurado=89, n_censurado=11",
             fontsize=12, fontweight="bold")

# RMSE por grupo y modelo
model_names  = list(preds.keys())
grupos_label = ["Global", "No censurado\n(n=89)", "Censurado\n(n=11)"]
x = np.arange(len(model_names)); w = 0.26
colors_g = ["#546E7A", "#1976D2", "#E53935"]

ax = axes[0]
for gi, (gkey, gcolor) in enumerate(zip(
    [f"Global (n=100)", f"No censurado (n=89)", f"Censurado RUL≥125 (n=11)"],
    colors_g)):
    rmse_vals = [df_m2[(df_m2["Modelo"]==m)&(df_m2["Grupo"]==gkey)]["RMSE"].values[0]
                 for m in model_names]
    ax.bar(x + (gi-1)*w, rmse_vals, w, label=grupos_label[gi], color=gcolor, alpha=0.8)

ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=10)
ax.set_ylabel("RMSE (ciclos)"); ax.set_title("RMSE por grupo de censura y modelo")
ax.legend(fontsize=8.5)

# Scatter censurado vs no censurado (XGBoost)
ax2 = axes[1]
nc_mask = ~censurado
ax2.scatter(y_test[nc_mask], y_pred_xgb[nc_mask],
            c="#1976D2", s=45, alpha=0.7, label="No censurado (n=89)", zorder=3)
ax2.scatter(y_test[censurado], y_pred_xgb[censurado],
            c="#E53935", s=80, alpha=0.9, marker="D",
            label="Censurado — target truncado a 125 (n=11)", zorder=4)
lims = [0, RUL_CAP]
ax2.plot(lims, lims, "k--", lw=1.2, label="Predicción perfecta")
ax2.fill_between(lims, [l-15 for l in lims], [l+15 for l in lims],
                 alpha=0.1, color="gray")
ax2.set_xlabel("RUL target (ciclos, cap=125)"); ax2.set_ylabel("RUL predicho (ciclos)")
ax2.set_title("XGBoost — predicho vs real\n(rombos rojos: error contaminado por censura)")
ax2.legend(fontsize=8.5)
# Anotar motores censurados
for i, (yt, yp) in enumerate(zip(y_test[censurado], y_pred_xgb[censurado])):
    ax2.annotate(f"↑{rul_true_raw[censurado][i]}", (yt, yp),
                 fontsize=6.5, color="#B71C1C",
                 xytext=(2, 3), textcoords="offset points")

plt.tight_layout()
p_m2 = f"{OUTPUT_DIR}/f7_02_censura.png"
fig.savefig(p_m2, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m2}")


# ════════════════════════════════════════════════════════════════════════════
# MEJORA 3 — PHM Score asimétrica (Saxena et al. 2008)
# ════════════════════════════════════════════════════════════════════════════
print("\n[M3] PHM score asimétrica…")

def phm_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return s.sum()

def phm_per_motor(y_true, y_pred):
    d = y_pred - y_true
    return np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)

phm_scores = {name: phm_score(y_test, yp) for name, yp in preds.items()}
rmse_scores = {name: np.sqrt(np.mean((yp-y_test)**2)) for name, yp in preds.items()}

print("  PHM scores (menor = mejor):")
for name, s in sorted(phm_scores.items(), key=lambda x: x[1]):
    print(f"    {name:15s}: {s:8.1f}  (RMSE={rmse_scores[name]:.2f})")

# Ranking por RMSE vs PHM
rank_rmse = pd.Series(rmse_scores).rank()
rank_phm  = pd.Series(phm_scores).rank()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Mejora 3 — PHM Score asimétrica (Saxena et al. 2008)\n"
             "Penalización asimétrica: sobreestimar RUL es más costoso que subestimar",
             fontsize=12, fontweight="bold")

# Panel 1: PHM score por modelo
ax = axes[0]
model_names_sorted = sorted(phm_scores, key=lambda x: phm_scores[x])
vals = [phm_scores[m] for m in model_names_sorted]
bar_c = ["#EF5350","#FF7043","#66BB6A","#42A5F5"]
bars = ax.bar(model_names_sorted, vals, color=bar_c, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+20, f"{v:.0f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("PHM Score (suma — menor=mejor)")
ax.set_title("PHM Score por modelo\n(exp asimétrico: α_late=13, α_early=10)")

# Panel 2: Distribución PHM per-motor (XGBoost vs LightGBM)
ax2 = axes[1]
phm_xgb = phm_per_motor(y_test, y_pred_xgb)
phm_lgb = phm_per_motor(y_test, y_pred_lgb)
ax2.scatter(range(n), phm_xgb, s=20, alpha=0.6, color="#EF5350", label=f"XGB (Σ={phm_scores['XGBoost']:.0f})")
ax2.scatter(range(n), phm_lgb, s=20, alpha=0.6, color="#42A5F5", label=f"LGB (Σ={phm_scores['LightGBM']:.0f})")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlabel("Motor"); ax2.set_ylabel("PHM score por motor")
ax2.set_title("PHM score por motor\nXGBoost vs LightGBM")
ax2.legend(fontsize=8.5)

# Panel 3: Ranking RMSE vs PHM
ax3 = axes[2]
models_all = list(preds.keys())
x3 = np.arange(len(models_all)); w3 = 0.38
b1 = ax3.bar(x3-w3/2, [rank_rmse[m] for m in models_all], w3,
             label="Ranking RMSE", color="#1976D2", alpha=0.8)
b2 = ax3.bar(x3+w3/2, [rank_phm[m] for m in models_all], w3,
             label="Ranking PHM", color="#E65100", alpha=0.8)
ax3.set_xticks(x3); ax3.set_xticklabels(models_all, rotation=10)
ax3.set_ylabel("Posición (1=mejor)"); ax3.set_title("Cambio de ranking: RMSE vs PHM score")
ax3.set_yticks([1,2,3,4])
ax3.legend()
# Flechas indicando cambio
for i, m in enumerate(models_all):
    dr = rank_phm[m] - rank_rmse[m]
    if abs(dr) >= 1:
        ax3.annotate(f"Δ{dr:+.0f}", xy=(i+w3/2, rank_phm[m]),
                     xytext=(i+w3/2+0.05, rank_phm[m]-0.3),
                     fontsize=8, color="#B71C1C" if dr>0 else "#2E7D32",
                     fontweight="bold")

plt.tight_layout()
p_m3 = f"{OUTPUT_DIR}/f7_03_phm_score.png"
fig.savefig(p_m3, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m3}")


# ════════════════════════════════════════════════════════════════════════════
# MEJORA 4 — SHAP tree_path_dependent + ablación s14
# ════════════════════════════════════════════════════════════════════════════
print("\n[M4] SHAP tree_path_dependent + ablación s14…")

# 4a. SHAP con tree_path_dependent
explainer_tpd = shap.TreeExplainer(xgb_model,
                                   feature_perturbation="tree_path_dependent")
shap_tpd = explainer_tpd.shap_values(X_test)   # (100, 112)

mean_shap_tpd = pd.Series(np.abs(shap_tpd).mean(axis=0), index=feat_cols)

# También recalcular interventional (ya lo teníamos pero con X_bg distinto, rehacemos)
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_train), min(500, len(X_train)), replace=False)
explainer_int = shap.TreeExplainer(xgb_model, data=X_train[bg_idx],
                                   feature_perturbation="interventional")
shap_int = explainer_int.shap_values(X_test)
mean_shap_int = pd.Series(np.abs(shap_int).mean(axis=0), index=feat_cols)

# Comparar top-15
top15_tpd = mean_shap_tpd.sort_values(ascending=False).head(15)
top15_int = mean_shap_int.sort_values(ascending=False).head(15)

print("  Top 5 SHAP tree_path_dependent:")
for f, v in top15_tpd.head(5).items():
    print(f"    {f:30s} = {v:.3f}")
print("  Top 5 SHAP interventional:")
for f, v in top15_int.head(5).items():
    print(f"    {f:30s} = {v:.3f}")

# 4b. Ablación: XGBoost sin features de s14
INFO_no14 = [s for s in INFO if s != "s14"]
X_train_no14, y_train_no14, groups_no14, fc_no14, fsc_no14, gsc_no14 = build_train(INFO_no14, WINDOWS)
X_test_no14, y_test_no14, _, _ = build_test(fsc_no14, gsc_no14, INFO_no14, WINDOWS)

xgb_no14 = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                              tree_method="hist", device="cpu", verbosity=0, random_state=42)
gkf = GroupKFold(n_splits=5)
cv_rmse_no14 = -cross_val_score(xgb_no14, X_train_no14, y_train_no14, groups=groups_no14,
                                 cv=gkf, scoring="neg_root_mean_squared_error", n_jobs=-1)
xgb_no14.fit(X_train_no14, y_train_no14)
y_pred_no14 = xgb_no14.predict(X_test_no14).clip(0, RUL_CAP)
rmse_no14   = np.sqrt(np.mean((y_pred_no14 - y_test)**2))

print(f"\n  Ablación s14:")
print(f"    CV RMSE  sin s14 = {cv_rmse_no14.mean():.3f} ± {cv_rmse_no14.std():.3f}")
print(f"    CV RMSE  con s14 = 12.925 (original)")
print(f"    Test RMSE sin s14 = {rmse_no14:.3f}")
print(f"    Test RMSE con s14 = 12.804 (original)")
print(f"    Δ RMSE = {rmse_no14-12.804:.3f} ciclos")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Mejora 4 — SHAP: tree_path_dependent vs interventional + ablación de s14",
             fontsize=12, fontweight="bold")

# Comparar importancias top-15
ax = axes[0]
# Fusionar en DataFrame de comparación
all_feats = list(set(top15_tpd.index) | set(top15_int.index))
comp = pd.DataFrame({
    "TPD":  mean_shap_tpd[all_feats],
    "INT":  mean_shap_int[all_feats],
}).sort_values("TPD", ascending=False).head(15)

x4 = np.arange(len(comp)); w4 = 0.38
ax.bar(x4-w4/2, comp["TPD"], w4, label="tree_path_dependent", color="#7B1FA2", alpha=0.8)
ax.bar(x4+w4/2, comp["INT"], w4, label="interventional",       color="#0288D1", alpha=0.8)
ax.set_xticks(x4); ax.set_xticklabels(comp.index, rotation=40, ha="right", fontsize=7.5)
ax.set_ylabel("Mean |SHAP| (ciclos)"); ax.legend(fontsize=8.5)
ax.set_title("Importancia SHAP: dos métodos de perturbación\n(divergencias = efecto de multicolinealidad s9-s14)")

# Destaque de features s9/s14 (en ambos métodos)
for i, feat in enumerate(comp.index):
    if "s9" in feat or "s14" in feat:
        ax.axvspan(i-0.5, i+0.5, alpha=0.08, color="#E65100")
ax.text(0.02, 0.97, "Naranja: features s9/s14 (r=0.963)",
        transform=ax.transAxes, fontsize=8, va="top", style="italic",
        bbox=dict(boxstyle="round", facecolor="#FFF3E0", alpha=0.8))

# Ablación s14
ax2 = axes[1]
ablation_data = {
    "Con s14\n(original)":   {"CV RMSE": 12.925, "Test RMSE": 12.804},
    "Sin s14\n(ablación)":   {"CV RMSE": cv_rmse_no14.mean(), "Test RMSE": rmse_no14},
}
x5 = np.arange(2); w5 = 0.38
cv_vals   = [ablation_data[k]["CV RMSE"]   for k in ablation_data]
test_vals = [ablation_data[k]["Test RMSE"] for k in ablation_data]
b1 = ax2.bar(x5-w5/2, cv_vals,   w5, label="CV RMSE",   color="#1565C0", alpha=0.7, hatch="///")
b2 = ax2.bar(x5+w5/2, test_vals, w5, label="Test RMSE", color="#1565C0", alpha=0.9)
ax2.set_xticks(x5); ax2.set_xticklabels(list(ablation_data.keys()), fontsize=10)
ax2.set_ylabel("RMSE (ciclos)")
ax2.set_title(f"Ablación de s14 en XGBoost\nΔ Test RMSE = {rmse_no14-12.804:+.3f} ciclos")
ax2.legend()
ax2.set_ylim(10, 16)
for bar, val in zip(list(b1)+list(b2), cv_vals+test_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.05, f"{val:.3f}",
             ha="center", fontsize=9, fontweight="bold")
conclusion = (f"s14 {'aporta' if rmse_no14 > 12.804+0.5 else 'es prescindible — redundante con s9'}")
ax2.text(0.5, 0.05, conclusion, ha="center", va="bottom", transform=ax2.transAxes,
         fontsize=9, style="italic",
         bbox=dict(boxstyle="round", facecolor="#E8F5E9" if "prescindible" in conclusion else "#FFEBEE", alpha=0.8))

plt.tight_layout()
p_m4 = f"{OUTPUT_DIR}/f7_04_shap_ablacion.png"
fig.savefig(p_m4, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m4}")


# ════════════════════════════════════════════════════════════════════════════
# MEJORA 5 — Durbin-Watson y Breusch-Pagan sobre trayectorias completas
# ════════════════════════════════════════════════════════════════════════════
print("\n[M5] Durbin-Watson y Breusch-Pagan…")

# Construir trayectorias completas de test (todos los ciclos de cada motor)
gsc_m5 = joblib.load(f"{MODEL_DIR}/global_sensor_scaler.pkl")
fsc_m5 = joblib.load(f"{MODEL_DIR}/final_scaler.pkl")
test_raw = load_raw("test"); test_raw[INFO] = gsc_m5.transform(test_raw[INFO].values)
test_full = add_rolling(test_raw, INFO, WINDOWS)
rul_df2   = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", header=None, names=["RUL_true"])

def dw_stat(residuals):
    """Estadístico Durbin-Watson."""
    d = np.diff(residuals)
    return np.sum(d**2) / np.sum(residuals**2)

dw_results = []
all_residuals, all_rul_true = [], []

for eid in range(1, 101):
    eng = test_full[test_full["engine_id"]==eid].sort_values("cycle")
    n_cyc = len(eng)
    rul_true_last = rul_df2.loc[eid-1, "RUL_true"]
    # RUL real a lo largo de la trayectoria
    rul_real = np.arange(n_cyc + rul_true_last, rul_true_last, -1)[:n_cyc].clip(0, RUL_CAP)

    Xeng = np.nan_to_num(eng[feat_cols].values.astype(np.float32), nan=0.0)
    Xeng = fsc_m5.transform(Xeng)
    yp   = xgb_model.predict(Xeng).clip(0, RUL_CAP)
    res  = yp - rul_real

    dw = dw_stat(res)
    dw_results.append({"engine_id": eid, "DW": dw, "n_cycles": n_cyc,
                        "mean_error": res.mean(), "std_error": res.std()})
    all_residuals.append(res)
    all_rul_true.append(rul_real)

dw_df = pd.DataFrame(dw_results)
all_res_flat = np.concatenate(all_residuals)
all_rul_flat = np.concatenate(all_rul_true)

print(f"  DW medio por motor: {dw_df['DW'].mean():.3f} ± {dw_df['DW'].std():.3f}")
print(f"  Motores con DW < 1.5 (autocorrelación positiva): {(dw_df['DW']<1.5).sum()}")
print(f"  Motores con DW > 2.5 (autocorrelación negativa): {(dw_df['DW']>2.5).sum()}")

# Breusch-Pagan sobre residuos vs RUL (heteroscedasticidad)
# Regresión: |residual| ~ RUL
from scipy.stats import f as fdist
bp_res = np.abs(all_res_flat)
X_bp   = np.column_stack([np.ones(len(all_rul_flat)), all_rul_flat])
beta   = np.linalg.lstsq(X_bp, bp_res, rcond=None)[0]
fitted = X_bp @ beta
ss_reg = np.sum((fitted - bp_res.mean())**2)
ss_tot = np.sum((bp_res - bp_res.mean())**2)
r2_bp  = ss_reg / ss_tot
n_bp   = len(bp_res); k_bp = 1
F_bp   = (r2_bp / k_bp) / ((1 - r2_bp) / (n_bp - k_bp - 1))
p_bp   = 1 - fdist.cdf(F_bp, k_bp, n_bp - k_bp - 1)
print(f"  Breusch-Pagan F={F_bp:.2f}  p={p_bp:.4f}  → {'Heteroscedasticidad' if p_bp<0.05 else 'Homocedasticidad'}")
print(f"  β_RUL = {beta[1]:.4f} (si negativo: mayor error en zona de RUL alto)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Mejora 5 — Análisis de residuos: autocorrelación y heteroscedasticidad\n"
             "XGBoost | Trayectorias completas de test (100 motores)",
             fontsize=12, fontweight="bold")

# Panel 1: Histograma DW
ax = axes[0,0]
ax.hist(dw_df["DW"], bins=20, color="#1976D2", alpha=0.75, edgecolor="white")
ax.axvline(1.5, color="#E53935", lw=2, linestyle="--", label="DW=1.5 (umbral)")
ax.axvline(2.0, color="#43A047", lw=2, linestyle="--", label="DW=2.0 (sin autocorr.)")
ax.axvline(2.5, color="#E53935", lw=2, linestyle="--", label="DW=2.5")
ax.axvline(dw_df["DW"].mean(), color="black", lw=2, label=f"Media={dw_df['DW'].mean():.2f}")
ax.set_xlabel("Estadístico Durbin-Watson"); ax.set_ylabel("Frecuencia")
ax.set_title("Distribución DW por motor\n(1.5–2.5 = sin autocorrelación problemática)")
ax.legend(fontsize=7.5)

# Panel 2: DW vs longitud de trayectoria
ax2 = axes[0,1]
sc = ax2.scatter(dw_df["n_cycles"], dw_df["DW"],
                 c=dw_df["DW"], cmap="RdYlGn", vmin=1.0, vmax=3.0, s=40, alpha=0.8)
ax2.axhline(1.5, color="#E53935", lw=1.5, linestyle="--")
ax2.axhline(2.5, color="#E53935", lw=1.5, linestyle="--")
ax2.set_xlabel("Ciclos de trayectoria"); ax2.set_ylabel("Durbin-Watson")
ax2.set_title("DW vs longitud de trayectoria de test")
plt.colorbar(sc, ax=ax2, label="DW")

# Panel 3: Residuos vs RUL real (heteroscedasticidad)
ax3 = axes[1,0]
ax3.scatter(all_rul_flat, all_res_flat, s=3, alpha=0.3, color="#546E7A")
# Línea suavizada (media rolling)
rul_sort_idx = np.argsort(all_rul_flat)
rul_sorted   = all_rul_flat[rul_sort_idx]
res_sorted   = all_res_flat[rul_sort_idx]
smooth_mean  = pd.Series(res_sorted).rolling(500, center=True, min_periods=1).mean().values
smooth_std   = pd.Series(np.abs(res_sorted)).rolling(500, center=True, min_periods=1).mean().values
ax3.plot(rul_sorted, smooth_mean, color="#E53935", lw=2, label="Media residuos (rolling)")
ax3.fill_between(rul_sorted, smooth_mean-smooth_std, smooth_mean+smooth_std,
                 alpha=0.2, color="#E53935", label="±MAE (rolling)")
ax3.axhline(0, color="black", lw=1)
ax3.set_xlabel("RUL real (ciclos)"); ax3.set_ylabel("Residuo (predicho − real)")
ax3.set_title(f"Residuos vs RUL real\nBreusch-Pagan F={F_bp:.1f}  p={'<0.001' if p_bp<0.001 else f'{p_bp:.4f}'}")
ax3.legend(fontsize=8)

# Panel 4: Autocorrelación de residuos (ACF manual, media entre motores)
ax4 = axes[1,1]
max_lag = 20
acf_vals = []
for lag in range(1, max_lag+1):
    acfs_motor = []
    for res in all_residuals:
        if len(res) > lag+5:
            r, _ = pearsonr(res[:-lag], res[lag:])
            acfs_motor.append(r)
    acf_vals.append(np.mean(acfs_motor))

ci = 1.96 / np.sqrt(np.mean([len(r) for r in all_residuals]))
lags = range(1, max_lag+1)
ax4.bar(lags, acf_vals, color=["#E53935" if abs(v)>ci else "#1976D2" for v in acf_vals],
        alpha=0.8)
ax4.axhline(ci, color="#E53935", lw=1.2, linestyle="--", label=f"±IC 95% ({ci:.3f})")
ax4.axhline(-ci, color="#E53935", lw=1.2, linestyle="--")
ax4.axhline(0, color="black", lw=0.8)
ax4.set_xlabel("Lag (ciclos)"); ax4.set_ylabel("Autocorrelación media")
ax4.set_title("Función de Autocorrelación (ACF)\nmedia entre motores — residuos XGBoost")
ax4.legend(fontsize=8.5)
sig_lags = [l for l, v in zip(lags, acf_vals) if abs(v) > ci]
if sig_lags:
    ax4.text(0.98, 0.97, f"Lags significativos: {sig_lags}",
             ha="right", va="top", transform=ax4.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.8))

plt.tight_layout()
p_m5 = f"{OUTPUT_DIR}/f7_05_residuos.png"
fig.savefig(p_m5, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m5}")


# ════════════════════════════════════════════════════════════════════════════
# MEJORA 6 — Ablación de ventanas temporales (LightGBM)
# ════════════════════════════════════════════════════════════════════════════
print("\n[M6] Ablación de ventanas temporales (LightGBM)…")

lgb_params = dict(n_estimators=200, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                  verbose=-1, random_state=42, n_jobs=-1)

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
# Línea referencia
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

# Línea de Pareto
pareto_x = abl_df["n_feats"].values
pareto_y = abl_df["CV_RMSE"].values
ax2.plot(pareto_x, pareto_y, "gray", alpha=0.3, lw=1, linestyle="--")

plt.tight_layout()
p_m6 = f"{OUTPUT_DIR}/f7_06_ablacion_ventanas.png"
fig.savefig(p_m6, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Guardado: {p_m6}")


# ════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RESUMEN MEJORAS ESTADÍSTICAS — Fase 7")
print("="*65)
print(f"\nM1 (Diebold-Mariano): XGB vs LGB  p={p_hln:.4f}  → "
      f"{'No significativo — equivalentes' if p_hln>0.05 else 'Significativo — XGB mejor'}")
print(f"M2 (Censura):  RMSE no censurado={metricas(y_test[~censurado], y_pred_xgb[~censurado])['RMSE']:.3f}  "
      f"RMSE censurado={metricas(y_test[censurado], y_pred_xgb[censurado])['RMSE']:.3f}")
print(f"M3 (PHM):  {sorted(phm_scores.items(), key=lambda x:x[1])}")
print(f"M4 (SHAP+s14):  Δ RMSE sin s14 = {rmse_no14-12.804:+.3f} ciclos")
print(f"M5 (DW):  DW medio={dw_df['DW'].mean():.3f}  BP p={p_bp:.4f}")
print(f"M6 (Ventanas):  Mejor combo={abl_df.iloc[0]['Ventanas']} CV={abl_df.iloc[0]['CV_RMSE']:.3f}")

# Guardar tabla M6
abl_df.to_csv(f"{OUTPUT_DIR}/fase7_ablacion_ventanas.csv", index=False)
print("\n[FASE 7 COMPLETADA]")
