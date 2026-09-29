#!/usr/bin/env python
"""Pipeline único y reproducible -- NASA C-MAPSS FD001, predicción de RUL.

Ejecuta de principio a fin: carga -> escalado global (fit solo en train) ->
ablación de ventanas por CV -> ablación de s14 por CV -> búsqueda de
hiperparámetros por CV -> selección de modelo por CV -> UN solo toque al
test set para las métricas finales -> bloque estadístico corregido ->
guardado de modelos y figuras.

Uso:
    python train.py

Sustituye a scripts/fase1..fase7 (que tenían tres definiciones de features
incompatibles entre sí y decisiones de modelado tomadas mirando el test set;
ver el historial de commits de este repositorio para el detalle de la corrección).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cmapss import config, data, features, modeling, stats_eval  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
np.random.seed(config.SEED)

plt.rcParams.update({
    "figure.facecolor": config.PALETTE["background"],
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": config.PALETTE["text"],
    "text.color": config.PALETTE["text"],
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA Y ESCALADO GLOBAL (fit SOLO en train -- corrige el leakage
#    intra-motor de normalize_per_unit, que ajustaba MinMax por engine_id
#    sobre la trayectoria COMPLETA, incluyendo test)
# ═══════════════════════════════════════════════════════════════════════════
log("Cargando datos crudos FD001...")
train_df, test_df, rul_df = data.load_cmapss()
log(f"  train={train_df.shape}  test={test_df.shape}  rul={rul_df.shape}")

informative, low_var = data.select_informative_sensors(train_df, test_df)
log(f"  sensores informativos ({len(informative)}): {informative}")
log(f"  descartados por varianza nula ({len(low_var)}): {low_var}")

global_scaler = MinMaxScaler()
train_df[informative] = global_scaler.fit_transform(train_df[informative])
test_df[informative] = global_scaler.transform(test_df[informative])
joblib.dump(global_scaler, config.MODEL_DIR / "global_sensor_scaler.pkl")
log("  MinMaxScaler global ajustado SOLO en train, aplicado a train y test "
    f"(n_samples_seen_={global_scaler.n_samples_seen_})")

train_rul = data.compute_rul_train(train_df)
test_rul = data.compute_rul_test(test_df, rul_df)
n_censored = int(test_rul["censored"].sum())
log(f"  RUL cap={config.RUL_CAP} ciclos | motores de test censurados (RUL_true>=cap): {n_censored}/100")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ABLACIÓN DE VENTANAS TEMPORALES -- por CV, con un modelo sonda fijo
#    (LightGBM, hiperparámetros modestos). El tamaño de ventana NO se elige
#    a mano ni se justifica con el test set (auditoría 5.9).
# ═══════════════════════════════════════════════════════════════════════════
import lightgbm as lgb  # noqa: E402

PROBE_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                     random_state=config.SEED, verbose=-1, n_jobs=-1)


def cv_rmse_groupkfold(X, y, groups, model_factory, n_splits=config.N_OUTER_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = model_factory()
        model.fit(X[tr_idx], y[tr_idx])
        preds = modeling.clip_predictions(model.predict(X[val_idx]))
        scores.append(rmse(y[val_idx], preds))
    return np.array(scores)


log("Ablación de ventanas temporales (modelo sonda LightGBM, GroupKFold k=5)...")
window_results = []
for windows in config.WINDOW_CANDIDATES:
    feat = features.build_features(train_rul, informative, windows)
    cols = features.feature_columns(informative, windows)
    X = feat[cols].to_numpy(dtype=np.float32)
    y = feat["RUL"].to_numpy(dtype=np.float32)
    groups = feat["engine_id"].to_numpy()
    scores = cv_rmse_groupkfold(X, y, groups, lambda: lgb.LGBMRegressor(**PROBE_PARAMS))
    window_results.append({
        "windows": str(windows), "n_features": len(cols),
        "cv_rmse_mean": scores.mean(), "cv_rmse_std": scores.std(),
    })
    log(f"  windows={windows!s:<16} n_features={len(cols):<4} "
        f"CV RMSE = {scores.mean():.3f} +/- {scores.std():.3f}")

window_df = pd.DataFrame(window_results).sort_values("cv_rmse_mean").reset_index(drop=True)
window_df.to_csv(config.OUTPUT_DIR / "ablacion_ventanas.csv", index=False)

best_row = window_df.iloc[0]
tolerance = best_row["cv_rmse_std"]
within_tol = window_df[window_df["cv_rmse_mean"] <= best_row["cv_rmse_mean"] + tolerance]
chosen_row = within_tol.sort_values("n_features").iloc[0]
WINDOWS = eval(chosen_row["windows"])
log(f"  -> Ventanas elegidas por CV: {WINDOWS} "
    f"(RMSE={chosen_row['cv_rmse_mean']:.3f}+/-{chosen_row['cv_rmse_std']:.3f}, "
    f"{'ganadora directa' if chosen_row['windows'] == best_row['windows'] else 'empate con la ganadora dentro de 1 std, se prefiere la más parsimoniosa'})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ABLACIÓN DE s14 -- mismo pipeline y modelo sonda para los dos brazos,
#    decisión por CV (auditoría 5.2/5.3: antes se comparaba un brazo
#    entrenado con datos/params distintos, y la decisión se leía del test).
# ═══════════════════════════════════════════════════════════════════════════
log("Ablación del sensor s14 (mismo pipeline en ambos brazos, GroupKFold k=5)...")
sensors_with_s14 = informative
sensors_without_s14 = [s for s in informative if s != "s14"]

feat_with = features.build_features(train_rul, sensors_with_s14, WINDOWS)
cols_with = features.feature_columns(sensors_with_s14, WINDOWS)
X_with = feat_with[cols_with].to_numpy(dtype=np.float32)
y_with = feat_with["RUL"].to_numpy(dtype=np.float32)
groups_with = feat_with["engine_id"].to_numpy()
scores_with = cv_rmse_groupkfold(X_with, y_with, groups_with, lambda: lgb.LGBMRegressor(**PROBE_PARAMS))

feat_without = features.build_features(train_rul, sensors_without_s14, WINDOWS)
cols_without = features.feature_columns(sensors_without_s14, WINDOWS)
X_without = feat_without[cols_without].to_numpy(dtype=np.float32)
y_without = feat_without["RUL"].to_numpy(dtype=np.float32)
groups_without = feat_without["engine_id"].to_numpy()
scores_without = cv_rmse_groupkfold(X_without, y_without, groups_without, lambda: lgb.LGBMRegressor(**PROBE_PARAMS))

log(f"  con s14    : CV RMSE = {scores_with.mean():.3f} +/- {scores_with.std():.3f}")
log(f"  sin s14    : CV RMSE = {scores_without.mean():.3f} +/- {scores_without.std():.3f}")

t_s14, p_s14 = stats_eval.paired_ttest(scores_with, scores_without)
keep_s14 = scores_with.mean() <= scores_without.mean()
log(f"  t-test pareado por fold: t={t_s14:.3f}, p={p_s14:.4f} "
    f"-> decisión (por CV): {'CONSERVAR s14' if keep_s14 else 'ELIMINAR s14 (redundante con s9, r=0.963)'}")

pd.DataFrame([
    {"variante": "con_s14", "cv_rmse_mean": scores_with.mean(), "cv_rmse_std": scores_with.std()},
    {"variante": "sin_s14", "cv_rmse_mean": scores_without.mean(), "cv_rmse_std": scores_without.std()},
]).to_csv(config.OUTPUT_DIR / "ablacion_s14.csv", index=False)

FINAL_SENSORS = sensors_with_s14 if keep_s14 else sensors_without_s14
FINAL_COLS = features.feature_columns(FINAL_SENSORS, WINDOWS)
log(f"  Feature set final: {len(FINAL_SENSORS)} sensores x ventanas {WINDOWS} -> {len(FINAL_COLS)} features")


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DEL FEATURE SET FINAL (train completo + test completo,
#    para poder reconstruir trayectorias de degradación en el bloque
#    estadístico) y último ciclo de test (evaluación oficial).
# ═══════════════════════════════════════════════════════════════════════════
train_feat = feat_with if keep_s14 else feat_without
test_feat_full = features.build_features(test_rul, FINAL_SENSORS, WINDOWS)

test_last = (test_feat_full.sort_values("cycle")
             .groupby("engine_id").tail(1).reset_index(drop=True))

X_train = train_feat[FINAL_COLS].to_numpy(dtype=np.float32)
y_train = train_feat["RUL"].to_numpy(dtype=np.float32)
groups_train = train_feat["engine_id"].to_numpy()

X_test = test_last[FINAL_COLS].to_numpy(dtype=np.float32)
y_test = test_last["RUL"].to_numpy(dtype=np.float32)
y_test_raw = test_last["RUL_raw"].to_numpy(dtype=np.float32)
censored_mask = test_last["censored"].to_numpy()

log(f"  X_train={X_train.shape}  X_test={X_test.shape} (1 fila/motor, 100 motores)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. BASELINES TRIVIALES (ausentes en el proyecto original)
# ═══════════════════════════════════════════════════════════════════════════
train_life = train_df.groupby("engine_id")["cycle"].max()
test_cycles_observed = test_df.groupby("engine_id")["cycle"].max().reindex(test_last["engine_id"]).to_numpy()
baselines = modeling.baseline_predictions(y_train, train_life.mean(), test_cycles_observed, len(y_test))


# ═══════════════════════════════════════════════════════════════════════════
# 6. BÚSQUEDA DE HIPERPARÁMETROS POR CV (RandomizedSearchCV + GroupKFold)
#    para los 4 modelos, sobre el feature set YA CONGELADO por las
#    ablaciones anteriores. Ningún hiperparámetro se fija "a mano".
# ═══════════════════════════════════════════════════════════════════════════
log(f"Búsqueda de hiperparámetros (RandomizedSearchCV, n_iter={config.N_SEARCH_ITER}, "
    f"GroupKFold k={config.N_SEARCH_FOLDS})...")

best_pipelines = {}
search_summary = []
for name in modeling.MODEL_NAMES:
    t_start = time.time()
    pipe = modeling.build_pipeline(name)
    gkf = GroupKFold(n_splits=config.N_SEARCH_FOLDS)
    search = RandomizedSearchCV(
        pipe, modeling.PARAM_DISTRIBUTIONS[name],
        n_iter=modeling.search_iter(name, config.N_SEARCH_ITER),
        scoring="neg_root_mean_squared_error",
        cv=gkf, random_state=config.SEED, n_jobs=modeling.outer_njobs(name), refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best_pipelines[name] = search.best_estimator_
    elapsed = time.time() - t_start
    log(f"  {name:<13} mejor CV RMSE(search)={-search.best_score_:.3f}  "
        f"params={search.best_params_}  ({elapsed:.0f}s)")
    search_summary.append({"modelo": name, "cv_rmse_search": -search.best_score_,
                            "mejores_params": json.dumps(search.best_params_, default=str)})

pd.DataFrame(search_summary).to_csv(config.OUTPUT_DIR / "busqueda_hiperparametros.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 7. PREDICCIONES OUT-OF-FOLD (GroupKFold k=5) con los hiperparámetros
#    elegidos -- ESTA es la CV que decide el modelo ganador, nunca el test.
# ═══════════════════════════════════════════════════════════════════════════
log("Prediciones out-of-fold (GroupKFold k=5) con hiperparámetros ya elegidos...")
gkf_outer = GroupKFold(n_splits=config.N_OUTER_FOLDS)
oof_preds = {}
oof_fold_rmse = {}
for name, pipe in best_pipelines.items():
    preds = cross_val_predict(pipe, X_train, y_train, groups=groups_train, cv=gkf_outer,
                               n_jobs=modeling.outer_njobs(name))
    preds = modeling.clip_predictions(preds)
    oof_preds[name] = preds
    fold_scores = []
    for _, val_idx in gkf_outer.split(X_train, y_train, groups_train):
        fold_scores.append(rmse(y_train[val_idx], preds[val_idx]))
    oof_fold_rmse[name] = np.array(fold_scores)
    log(f"  {name:<13} CV RMSE(OOF) = {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}  "
        f"R2(OOF) = {r2_score(y_train, preds):.4f}")

cv_ranking = sorted(oof_fold_rmse, key=lambda n: oof_fold_rmse[n].mean())
BEST_MODEL = cv_ranking[0]
RUNNER_UP = cv_ranking[1]
log(f"  -> Modelo elegido por CV: {BEST_MODEL} (runner-up: {RUNNER_UP})")


# ═══════════════════════════════════════════════════════════════════════════
# 8. COMPARACIONES ESTADÍSTICAS ENTRE MODELOS -- sobre las OOF de train,
#    NUNCA sobre el test. t-test pareado por motor (agregando el error
#    cuadrático OOF a nivel de motor -> 100 observaciones pareadas),
#    corrección de Holm sobre las 6 comparaciones, IC bootstrap.
# ═══════════════════════════════════════════════════════════════════════════
log("Comparaciones estadísticas entre modelos (OOF agregado por motor, Holm)...")
oof_df = pd.DataFrame({"engine_id": groups_train, "y_true": y_train})
for name, preds in oof_preds.items():
    oof_df[f"sqerr_{name}"] = (preds - y_train) ** 2
oof_by_engine = oof_df.groupby("engine_id")[[f"sqerr_{n}" for n in modeling.MODEL_NAMES]].mean()

pair_pvalues = {}
pair_records = []
from itertools import combinations
for a, b in combinations(modeling.MODEL_NAMES, 2):
    ea = oof_by_engine[f"sqerr_{a}"].to_numpy()
    eb = oof_by_engine[f"sqerr_{b}"].to_numpy()
    t, p = stats_eval.paired_ttest(ea, eb)
    ci = stats_eval.bootstrap_ci_diff_by_engine(ea, eb, seed=config.SEED)
    pair_pvalues[f"{a} vs {b}"] = p
    pair_records.append({"comparacion": f"{a} vs {b}", "t": t, "p": p,
                          "diff_mean_sqerr": ci["diff_mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"]})

holm_df = stats_eval.holm_correction(pair_pvalues)
pair_df = pd.DataFrame(pair_records).merge(holm_df[["comparacion", "p_holm", "significativo_holm_0.05"]], on="comparacion")
pair_df.to_csv(config.OUTPUT_DIR / "comparaciones_modelos_holm.csv", index=False)
log(f"  {len(pair_df)} comparaciones pareadas (t-test sobre error cuadrático medio por motor, OOF-CV); "
    f"{int(pair_df['significativo_holm_0.05'].sum())} significativas tras Holm (alfa=0.05)")


# ═══════════════════════════════════════════════════════════════════════════
# 9. REFIT FINAL SOBRE 100% DEL TRAIN Y ÚNICA EVALUACIÓN EN TEST
# ═══════════════════════════════════════════════════════════════════════════
log("Reentrenando modelos finales sobre 100% del train y evaluando en test (una sola vez)...")
final_models = {}
test_results = {}
for name, pipe in best_pipelines.items():
    pipe.fit(X_train, y_train)
    final_models[name] = pipe
    preds = modeling.clip_predictions(pipe.predict(X_test))
    test_results[name] = {
        "rmse": rmse(y_test, preds), "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds), "preds": preds,
        "phm": float(stats_eval.phm_score(y_test, preds).sum()),
    }
    # RMSE excluyendo los motores censurados (RUL_true >= cap): es una COTA
    # INFERIOR sobre el subconjunto no censurado, nunca "el RMSE real".
    unc = ~censored_mask
    test_results[name]["rmse_no_censurado_cota_inferior"] = rmse(y_test[unc], preds[unc])

for name, preds in baselines.items():
    test_results[name] = {
        "rmse": rmse(y_test, preds), "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds), "preds": preds,
        "phm": float(stats_eval.phm_score(y_test, preds).sum()),
        "rmse_no_censurado_cota_inferior": rmse(y_test[~censored_mask], preds[~censored_mask]),
    }

for name in modeling.MODEL_NAMES + list(baselines.keys()):
    r = test_results[name]
    log(f"  {name:<24} test RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}  R2={r['r2']:.4f}  PHM={r['phm']:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. GUARDADO DE MODELOS -- XGBoost en formato nativo .json (estable entre
#     versiones; el .pkl anterior era irrecuperable: RMSE=68 al recargar
#     por incompatibilidad binaria entre versiones de la librería).
# ═══════════════════════════════════════════════════════════════════════════
log("Guardando modelos y verificando que se recargan correctamente...")
for name, pipe in final_models.items():
    if name == "XGBoost":
        # XGBoost no usa scaler (passthrough); se guarda solo el booster en
        # formato nativo .json, estable entre versiones de la librería
        # (el .pkl del proyecto original era irrecuperable: RMSE=68 al
        # recargarlo por incompatibilidad binaria entre versiones).
        pipe.named_steps["model"].save_model(str(config.MODEL_DIR / "xgboost_fd001.json"))
    else:
        joblib.dump(pipe, config.MODEL_DIR / f"{name.lower()}_fd001.pkl")

# Verificación de humo: recargar y comprobar que reproduce el RMSE de test.
import xgboost as xgb  # noqa: E402
xgb_reloaded = xgb.XGBRegressor()
xgb_reloaded.load_model(str(config.MODEL_DIR / "xgboost_fd001.json"))
preds_reload = modeling.clip_predictions(xgb_reloaded.predict(X_test))
rmse_reload = rmse(y_test, preds_reload)
assert abs(rmse_reload - test_results["XGBoost"]["rmse"]) < 1e-3, (
    f"El modelo XGBoost recargado no reproduce el RMSE de test "
    f"({rmse_reload:.4f} vs {test_results['XGBoost']['rmse']:.4f})"
)
log(f"  [OK] XGBoost recargado reproduce RMSE de test = {rmse_reload:.4f}")

for name in ["Ridge", "RandomForest", "LightGBM"]:
    reloaded = joblib.load(config.MODEL_DIR / f"{name.lower()}_fd001.pkl")
    assert reloaded.named_steps["model"].n_features_in_ == X_test.shape[1], (
        f"{name}: n_features_in_ del modelo guardado no coincide con el feature set actual"
    )
    preds_reload = modeling.clip_predictions(reloaded.predict(X_test))
    assert abs(rmse(y_test, preds_reload) - test_results[name]["rmse"]) < 1e-3, (
        f"{name}: el modelo recargado no reproduce el RMSE de test"
    )
    log(f"  [OK] {name} recargado reproduce RMSE de test = {test_results[name]['rmse']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. BLOQUE ESTADÍSTICO SOBRE EL MODELO GANADOR -- trayectoria completa de
#     test, reconstrucción de RUL real SIN off-by-one, Ljung-Box y
#     Breusch-Pagan agregados por motor (no pseudo-replicados).
# ═══════════════════════════════════════════════════════════════════════════
log(f"Bloque estadístico sobre el modelo ganador ({BEST_MODEL}), trayectoria completa de test...")
best_final = final_models[BEST_MODEL]
X_test_full = test_feat_full[FINAL_COLS].to_numpy(dtype=np.float32)
preds_full = modeling.clip_predictions(best_final.predict(X_test_full))

residual_rows = []
for engine_id, g in test_feat_full.groupby("engine_id"):
    g = g.sort_values("cycle")
    n_cyc = len(g)
    rul_true_engine = float(rul_df.iloc[engine_id - 1]["RUL_true"])
    traj = stats_eval.reconstruct_true_rul_trajectory(n_cyc, rul_true_engine)
    preds_engine = preds_full[g.index]
    residual_rows.append(pd.DataFrame({
        "engine_id": engine_id, "cycle": g["cycle"].to_numpy(),
        "rul_true_traj": traj, "pred": preds_engine,
        "residual": traj - preds_engine,
    }))
residual_df = pd.concat(residual_rows, ignore_index=True)
residual_df.to_csv(config.OUTPUT_DIR / "residuos_trayectoria_completa.csv", index=False)

# Los contrastes de heterocedasticidad/autocorrelación se restringen a
# rul_true_traj <= RUL_CAP: el modelo se entrenó sobre el target CAPEADO, así
# que para ciclos con RUL real > 125 la predicción está mecánicamente topada
# en ~125 mientras el "residuo" sigue creciendo con el RUL real sin límite --
# eso no es heterocedasticidad del modelo, es el artefacto aritmético de
# comparar una predicción capeada contra un target sin capear. Se excluyen
# también los 11 motores censurados (su trayectoria completa queda por
# encima del cap). Sin este filtro, el propio cap generaba una tendencia
# casi perfectamente lineal en el tramo no capeado que dominaba e inflaba
# artificialmente el estadístico de Breusch-Pagan.
residual_capped = residual_df[residual_df["rul_true_traj"] <= config.RUL_CAP].copy()
log(f"  Filtrando a rul_true_traj<={config.RUL_CAP} para los contrastes de residuos: "
    f"{len(residual_capped)}/{len(residual_df)} ciclos, "
    f"{residual_capped['engine_id'].nunique()}/{residual_df['engine_id'].nunique()} motores "
    "(excluye el tramo donde el target de entrenamiento estaba capeado y los motores censurados)")

bp = stats_eval.heteroscedasticity_by_engine(residual_capped)
log(f"  Breusch-Pagan real (agregado por motor, n={bp['n_motores']}, solo RUL<=cap): "
    f"LM={bp['lm_stat']:.2f}, p={bp['lm_pvalue']:.2e}")

lb_df = stats_eval.ljung_box_by_engine(residual_capped, lags=10)
frac_autocorr = (lb_df["lb_pvalue"] < 0.05).mean()
log(f"  Ljung-Box (residuos centrados por motor, lag=10, solo RUL<=cap): "
    f"{frac_autocorr*100:.0f}% de los motores muestran autocorrelación significativa")

with open(config.OUTPUT_DIR / "estadistica_residuos.json", "w") as f:
    json.dump({
        "breusch_pagan_por_motor": bp,
        "ljung_box_frac_motores_autocorrelados": float(frac_autocorr),
        "nota": "Contrastes calculados solo sobre ciclos con rul_true_traj<=RUL_CAP "
                "(excluye el tramo capeado y los motores censurados); ver comentario en train.py",
        "n_ciclos_usados": int(len(residual_capped)),
        "n_ciclos_totales": int(len(residual_df)),
        "n_motores_usados": int(residual_capped["engine_id"].nunique()),
    }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 12. TABLA ÚNICA DE RESULTADOS (fuente de verdad para el README)
# ═══════════════════════════════════════════════════════════════════════════
log("Guardando tabla única de resultados...")
rows = []
for name in modeling.MODEL_NAMES:
    rows.append({
        "modelo": name,
        "cv_rmse_mean": oof_fold_rmse[name].mean(), "cv_rmse_std": oof_fold_rmse[name].std(),
        "test_rmse": test_results[name]["rmse"], "test_mae": test_results[name]["mae"],
        "test_r2": test_results[name]["r2"], "test_phm": test_results[name]["phm"],
        "test_rmse_no_censurado": test_results[name]["rmse_no_censurado_cota_inferior"],
    })
for name in baselines:
    rows.append({
        "modelo": name, "cv_rmse_mean": np.nan, "cv_rmse_std": np.nan,
        "test_rmse": test_results[name]["rmse"], "test_mae": test_results[name]["mae"],
        "test_r2": test_results[name]["r2"], "test_phm": test_results[name]["phm"],
        "test_rmse_no_censurado": test_results[name]["rmse_no_censurado_cota_inferior"],
    })
results_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
results_df.to_csv(config.OUTPUT_DIR / "resultados_finales.csv", index=False)
print(results_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# 13. FIGURAS
# ═══════════════════════════════════════════════════════════════════════════
log("Generando figuras...")


def savefig(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"  guardado {path.name}")


# f01: varianza de sensores
fig, ax = plt.subplots(figsize=(10, 6))
stds = train_df[data.SENSOR_COLS].std().sort_values()
colors = [config.PALETTE["danger"] if s in low_var else config.PALETTE["primary"] for s in stds.index]
ax.barh(stds.index, stds.values, color=colors)
ax.axvline(config.VARIANCE_THRESHOLD, color=config.PALETTE["danger"], linestyle="--", label="Umbral")
ax.set_xlabel("Desviación típica (sensor ya escalado a [0,1] globalmente)")
ax.set_title("Varianza de sensores -- FD001 train\nRojo = descartado (std < umbral)")
ax.legend()
savefig(fig, "f01_sensor_variance.png")

# f02: RUL raw vs capeado
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(train_rul["RUL_raw"], bins=50, color=config.PALETTE["neutral"])
axes[0].set_title("RUL crudo (sin cap)")
axes[1].hist(train_rul["RUL"], bins=50, color=config.PALETTE["primary"])
axes[1].set_title(f"RUL capeado a {config.RUL_CAP} ciclos")
for ax in axes:
    ax.set_xlabel("RUL (ciclos)")
fig.suptitle("Efecto del cap piecewise-linear del target (Heimes 2008)")
savefig(fig, "f02_rul_distribution.png")

# f03: ablación de ventanas
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(window_df["windows"], window_df["cv_rmse_mean"], yerr=window_df["cv_rmse_std"],
       color=config.PALETTE["primary"], capsize=4)
ax.axhline(chosen_row["cv_rmse_mean"], color=config.PALETTE["secondary"], linestyle="--",
           label=f"Elegida: {WINDOWS}")
ax.set_ylabel("CV RMSE (ciclos)")
ax.set_title("Ablación de ventanas temporales -- GroupKFold CV (modelo sonda LightGBM)")
ax.tick_params(axis="x", rotation=20)
ax.legend()
savefig(fig, "f03_ablacion_ventanas.png")

# f04: ablación s14
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["Con s14", "Sin s14"]
means = [scores_with.mean(), scores_without.mean()]
stds_ = [scores_with.std(), scores_without.std()]
ax.bar(labels, means, yerr=stds_, color=[config.PALETTE["primary"], config.PALETTE["accent"]], capsize=5)
ax.set_ylabel("CV RMSE (ciclos)")
ax.set_title(f"Ablación de s14 (mismo pipeline, GroupKFold CV)\n"
             f"t={t_s14:.2f}, p={p_s14:.3f} -> {'conservar' if keep_s14 else 'eliminar'}")
savefig(fig, "f04_ablacion_s14.png")

# f05: comparación CV entre modelos
fig, ax = plt.subplots(figsize=(9, 5))
names_sorted = cv_ranking
bp_data = [oof_fold_rmse[n] for n in names_sorted]
box = ax.boxplot(bp_data, tick_labels=names_sorted, patch_artist=True)
for patch, n in zip(box["boxes"], names_sorted):
    patch.set_facecolor(config.MODEL_COLORS[n])
    patch.set_alpha(0.7)
ax.set_ylabel("RMSE por fold (ciclos)")
ax.set_title(f"Comparación de modelos -- CV out-of-fold GroupKFold (k={config.N_OUTER_FOLDS})\n"
             f"Ganador por CV: {BEST_MODEL}")
savefig(fig, "f05_cv_comparison.png")

# f06: test set -- barras RMSE/PHM + pred vs real del ganador
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
order = results_df["modelo"].tolist()
axes[0].bar(order, results_df["test_rmse"], color=config.PALETTE["primary"])
axes[0].set_title("RMSE -- test oficial (100 motores)")
axes[0].tick_params(axis="x", rotation=60)
axes[1].bar(order, results_df["test_phm"], color=config.PALETTE["accent"])
axes[1].set_title("PHM score (Saxena 2008, menor=mejor)")
axes[1].tick_params(axis="x", rotation=60)
# Escala log: los baselines (PHM de miles/cientos de miles) aplastarían a
# escala lineal las barras de los 4 modelos reales (PHM de cientos) a algo
# visualmente indistinguible de 0 -- justo lo que hacía ilegible esta figura.
axes[1].set_yscale("log")
axes[1].set_ylabel("PHM score (escala log)")
best_preds = test_results[BEST_MODEL]["preds"]
order_idx = np.argsort(y_test)
axes[2].scatter(range(len(y_test)), y_test[order_idx], color=config.PALETTE["neutral"], label="RUL real", s=25)
axes[2].scatter(range(len(y_test)), best_preds[order_idx], color=config.PALETTE["primary"],
                label=f"{BEST_MODEL} (predicho)", s=25, alpha=0.8)
axes[2].fill_between(range(len(y_test)), y_test[order_idx] - 20, y_test[order_idx] + 20,
                      color=config.PALETTE["neutral"], alpha=0.12, label="+/-20 ciclos")
axes[2].set_title(f"Predicción vs real -- {BEST_MODEL}")
axes[2].legend(fontsize=8)
fig.suptitle("Evaluación única en el test set oficial (tocado una sola vez)")
savefig(fig, "f06_test_evaluation.png")

# f07: residuos vs RUL real (heteroscedasticidad). Se muestra la trayectoria
# completa por contexto, pero se marca en gris el tramo RUL>cap (donde la
# predicción está mecánicamente topada en ~125 mientras el eje X no lo está
# -- ese tramo NO entra en el Breusch-Pagan, que se calcula solo sobre
# RUL<=cap; mostrarlo sin distinguir haría parecer heterocedasticidad del
# modelo lo que en gran parte es aritmética del cap).
fig, ax = plt.subplots(figsize=(8, 5))
sample = residual_df.sample(min(5000, len(residual_df)), random_state=config.SEED)
above_cap = sample["rul_true_traj"] > config.RUL_CAP
ax.scatter(sample.loc[above_cap, "rul_true_traj"], sample.loc[above_cap, "residual"],
           s=6, alpha=0.15, color=config.PALETTE["neutral"],
           label=f"RUL>{config.RUL_CAP} (fuera del target de entrenamiento, excluido del test)")
ax.scatter(sample.loc[~above_cap, "rul_true_traj"], sample.loc[~above_cap, "residual"],
           s=6, alpha=0.3, color=config.PALETTE["primary"], label=f"RUL<={config.RUL_CAP} (usado en el test)")
ax.axvline(config.RUL_CAP, color=config.PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(0, color="black", linewidth=1)
ax.invert_xaxis()
ax.set_xlabel("RUL real (ciclos restantes)")
ax.set_ylabel("Residuo (RUL_real - RUL_pred)")
ax.legend(fontsize=7, loc="upper left")
ax.set_title(f"Residuos por ciclo -- {BEST_MODEL}, trayectoria completa de test\n"
             f"Breusch-Pagan (agregado por motor, n={bp['n_motores']}, solo RUL<={config.RUL_CAP}): "
             f"p={bp['lm_pvalue']:.1e}")
savefig(fig, "f07_residuals_heteroscedasticity.png")

# f08: SHAP summary (global) del modelo ganador -- sobre el test ya tocado,
# solo para interpretabilidad, no decide nada.
try:
    inner_model = best_final.named_steps["model"]
    explainer = shap.TreeExplainer(inner_model)
    shap_values = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, feature_names=FINAL_COLS, show=False, max_display=15)
    plt.title(f"SHAP summary -- {BEST_MODEL} (interpretación, no decide el pipeline)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "f08_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    log("  guardado f08_shap_summary.png")
except Exception as exc:  # pragma: no cover
    log(f"  [aviso] SHAP no disponible para {BEST_MODEL}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# 14. RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════
summary = {
    "sensores_informativos": informative,
    "sensores_descartados_por_varianza": low_var,
    "ventanas_elegidas_por_cv": WINDOWS,
    "s14_conservado_por_cv": bool(keep_s14),
    "n_features_final": len(FINAL_COLS),
    "modelo_ganador_por_cv": BEST_MODEL,
    "motores_test_censurados": n_censored,
    "tiempo_total_segundos": round(time.time() - T0, 1),
}
with open(config.OUTPUT_DIR / "resumen_pipeline.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

log("=" * 70)
log("PIPELINE COMPLETADO")
for k, v in summary.items():
    log(f"  {k}: {v}")
log("=" * 70)
