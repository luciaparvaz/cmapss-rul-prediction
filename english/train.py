#!/usr/bin/env python
"""Single, reproducible pipeline -- NASA C-MAPSS FD001, RUL prediction.

Runs end to end: load -> global scaling (fit on train only) -> window-size
ablation by CV -> s14 sensor ablation by CV -> hyperparameter search by CV
-> model selection by CV -> ONE single touch of the test set for final
metrics -> corrected statistical block -> model and figure saving.

Usage:
    python train.py

This is the English translation of the project's Spanish pipeline; same
logic, same formulas, same seeds. See the repository's commit history for
background on the methodology.
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
# 1. LOADING AND GLOBAL SCALING (fit ONLY on train -- avoids the intra-engine
#    leakage of per-unit normalization, which used to fit MinMax by
#    engine_id over the COMPLETE trajectory, including test)
# ═══════════════════════════════════════════════════════════════════════════
log("Loading raw FD001 data...")
train_df, test_df, rul_df = data.load_cmapss()
log(f"  train={train_df.shape}  test={test_df.shape}  rul={rul_df.shape}")

informative, low_var = data.select_informative_sensors(train_df, test_df)
log(f"  informative sensors ({len(informative)}): {informative}")
log(f"  dropped for near-zero variance ({len(low_var)}): {low_var}")

global_scaler = MinMaxScaler()
train_df[informative] = global_scaler.fit_transform(train_df[informative])
test_df[informative] = global_scaler.transform(test_df[informative])
joblib.dump(global_scaler, config.MODEL_DIR / "global_sensor_scaler.pkl")
log("  Global MinMaxScaler fit ONLY on train, applied to train and test "
    f"(n_samples_seen_={global_scaler.n_samples_seen_})")

train_rul = data.compute_rul_train(train_df)
test_rul = data.compute_rul_test(test_df, rul_df)
n_censored = int(test_rul["censored"].sum())
log(f"  RUL cap={config.RUL_CAP} cycles | censored test engines (RUL_true>=cap): {n_censored}/100")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL WINDOW ABLATION -- by CV, with a fixed probe model (LightGBM,
#    modest hyperparameters). Window size is NOT picked by hand or justified
#    with the test set.
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


log("Temporal window ablation (LightGBM probe model, GroupKFold k=5)...")
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
window_df.to_csv(config.OUTPUT_DIR / "window_ablation.csv", index=False)

best_row = window_df.iloc[0]
tolerance = best_row["cv_rmse_std"]
within_tol = window_df[window_df["cv_rmse_mean"] <= best_row["cv_rmse_mean"] + tolerance]
chosen_row = within_tol.sort_values("n_features").iloc[0]
WINDOWS = eval(chosen_row["windows"])
log(f"  -> Windows chosen by CV: {WINDOWS} "
    f"(RMSE={chosen_row['cv_rmse_mean']:.3f}+/-{chosen_row['cv_rmse_std']:.3f}, "
    f"{'direct winner' if chosen_row['windows'] == best_row['windows'] else 'tied with the winner within 1 std, the more parsimonious option is preferred'})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. s14 ABLATION -- same pipeline and probe model for both arms, decided by
#    CV (a prior version compared arms trained with different data/params,
#    and the decision was read off the test set).
# ═══════════════════════════════════════════════════════════════════════════
log("s14 sensor ablation (same pipeline in both arms, GroupKFold k=5)...")
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

log(f"  with s14   : CV RMSE = {scores_with.mean():.3f} +/- {scores_with.std():.3f}")
log(f"  without s14: CV RMSE = {scores_without.mean():.3f} +/- {scores_without.std():.3f}")

t_s14, p_s14 = stats_eval.paired_ttest(scores_with, scores_without)
keep_s14 = scores_with.mean() <= scores_without.mean()
log(f"  paired t-test per fold: t={t_s14:.3f}, p={p_s14:.4f} "
    f"-> decision (by CV): {'KEEP s14' if keep_s14 else 'DROP s14 (redundant with s9, r=0.963)'}")

pd.DataFrame([
    {"variant": "with_s14", "cv_rmse_mean": scores_with.mean(), "cv_rmse_std": scores_with.std()},
    {"variant": "without_s14", "cv_rmse_mean": scores_without.mean(), "cv_rmse_std": scores_without.std()},
]).to_csv(config.OUTPUT_DIR / "s14_ablation.csv", index=False)

FINAL_SENSORS = sensors_with_s14 if keep_s14 else sensors_without_s14
FINAL_COLS = features.feature_columns(FINAL_SENSORS, WINDOWS)
log(f"  Final feature set: {len(FINAL_SENSORS)} sensors x windows {WINDOWS} -> {len(FINAL_COLS)} features")


# ═══════════════════════════════════════════════════════════════════════════
# 4. BUILDING THE FINAL FEATURE SET (full train + full test, needed to
#    reconstruct degradation trajectories in the statistical block) and the
#    last test cycle (official evaluation).
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

log(f"  X_train={X_train.shape}  X_test={X_test.shape} (1 row/engine, 100 engines)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. TRIVIAL BASELINES (absent from the original project)
# ═══════════════════════════════════════════════════════════════════════════
train_life = train_df.groupby("engine_id")["cycle"].max()
test_cycles_observed = test_df.groupby("engine_id")["cycle"].max().reindex(test_last["engine_id"]).to_numpy()
baselines = modeling.baseline_predictions(y_train, train_life.mean(), test_cycles_observed, len(y_test))


# ═══════════════════════════════════════════════════════════════════════════
# 6. HYPERPARAMETER SEARCH BY CV (RandomizedSearchCV + GroupKFold) for the 4
#    models, on the feature set ALREADY FROZEN by the ablations above. No
#    hyperparameter is set "by hand".
# ═══════════════════════════════════════════════════════════════════════════
log(f"Hyperparameter search (RandomizedSearchCV, n_iter={config.N_SEARCH_ITER}, "
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
    log(f"  {name:<13} best CV RMSE(search)={-search.best_score_:.3f}  "
        f"params={search.best_params_}  ({elapsed:.0f}s)")
    search_summary.append({"model": name, "cv_rmse_search": -search.best_score_,
                            "best_params": json.dumps(search.best_params_, default=str)})

pd.DataFrame(search_summary).to_csv(config.OUTPUT_DIR / "hyperparameter_search.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 7. OUT-OF-FOLD PREDICTIONS (GroupKFold k=5) with the chosen hyperparameters
#    -- THIS is the CV that decides the winning model, never the test set.
# ═══════════════════════════════════════════════════════════════════════════
log("Out-of-fold predictions (GroupKFold k=5) with hyperparameters already chosen...")
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
log(f"  -> Model chosen by CV: {BEST_MODEL} (runner-up: {RUNNER_UP})")


# ═══════════════════════════════════════════════════════════════════════════
# 8. STATISTICAL COMPARISONS BETWEEN MODELS -- on the train OOF predictions,
#    NEVER on the test set. Paired t-test per engine (aggregating the OOF
#    squared error at the engine level -> 100 paired observations), Holm
#    correction over the 6 comparisons, bootstrap CI.
# ═══════════════════════════════════════════════════════════════════════════
log("Statistical comparisons between models (OOF aggregated by engine, Holm)...")
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
    pair_records.append({"comparison": f"{a} vs {b}", "t": t, "p": p,
                          "diff_mean_sqerr": ci["diff_mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"]})

holm_df = stats_eval.holm_correction(pair_pvalues)
pair_df = pd.DataFrame(pair_records).merge(holm_df[["comparison", "p_holm", "significant_holm_0.05"]], on="comparison")
pair_df.to_csv(config.OUTPUT_DIR / "model_comparisons_holm.csv", index=False)
log(f"  {len(pair_df)} paired comparisons (t-test on mean squared error per engine, OOF-CV); "
    f"{int(pair_df['significant_holm_0.05'].sum())} significant after Holm (alpha=0.05)")


# ═══════════════════════════════════════════════════════════════════════════
# 9. FINAL REFIT ON 100% OF TRAIN AND A SINGLE TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
log("Refitting final models on 100% of train and evaluating on test (once)...")
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
    # RMSE excluding censored engines (RUL_true >= cap): this is a LOWER
    # BOUND over the non-censored subset, never "the real RMSE".
    unc = ~censored_mask
    test_results[name]["rmse_uncensored_lower_bound"] = rmse(y_test[unc], preds[unc])

for name, preds in baselines.items():
    test_results[name] = {
        "rmse": rmse(y_test, preds), "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds), "preds": preds,
        "phm": float(stats_eval.phm_score(y_test, preds).sum()),
        "rmse_uncensored_lower_bound": rmse(y_test[~censored_mask], preds[~censored_mask]),
    }

for name in modeling.MODEL_NAMES + list(baselines.keys()):
    r = test_results[name]
    log(f"  {name:<24} test RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}  R2={r['r2']:.4f}  PHM={r['phm']:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MODEL SAVING -- XGBoost in native .json format (stable across library
#     versions; the previous .pkl was unrecoverable: RMSE=68 on reload due
#     to binary incompatibility between library versions).
# ═══════════════════════════════════════════════════════════════════════════
log("Saving models and verifying they reload correctly...")
for name, pipe in final_models.items():
    if name == "XGBoost":
        # XGBoost doesn't use a scaler (passthrough); only the booster is
        # saved, in the native .json format, stable across library versions
        # (the original project's .pkl was unrecoverable: RMSE=68 on reload
        # due to binary incompatibility between versions).
        pipe.named_steps["model"].save_model(str(config.MODEL_DIR / "xgboost_fd001.json"))
    else:
        joblib.dump(pipe, config.MODEL_DIR / f"{name.lower()}_fd001.pkl")

# Smoke test: reload and check it reproduces the test RMSE.
import xgboost as xgb  # noqa: E402
xgb_reloaded = xgb.XGBRegressor()
xgb_reloaded.load_model(str(config.MODEL_DIR / "xgboost_fd001.json"))
preds_reload = modeling.clip_predictions(xgb_reloaded.predict(X_test))
rmse_reload = rmse(y_test, preds_reload)
assert abs(rmse_reload - test_results["XGBoost"]["rmse"]) < 1e-3, (
    f"The reloaded XGBoost model does not reproduce the test RMSE "
    f"({rmse_reload:.4f} vs {test_results['XGBoost']['rmse']:.4f})"
)
log(f"  [OK] Reloaded XGBoost reproduces test RMSE = {rmse_reload:.4f}")

for name in ["Ridge", "RandomForest", "LightGBM"]:
    reloaded = joblib.load(config.MODEL_DIR / f"{name.lower()}_fd001.pkl")
    assert reloaded.named_steps["model"].n_features_in_ == X_test.shape[1], (
        f"{name}: the saved model's n_features_in_ does not match the current feature set"
    )
    preds_reload = modeling.clip_predictions(reloaded.predict(X_test))
    assert abs(rmse(y_test, preds_reload) - test_results[name]["rmse"]) < 1e-3, (
        f"{name}: the reloaded model does not reproduce the test RMSE"
    )
    log(f"  [OK] Reloaded {name} reproduces test RMSE = {test_results[name]['rmse']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. STATISTICAL BLOCK ON THE WINNING MODEL -- full test trajectory, true
#     RUL reconstruction WITHOUT the off-by-one, Ljung-Box and Breusch-Pagan
#     aggregated by engine (not pseudo-replicated).
# ═══════════════════════════════════════════════════════════════════════════
log(f"Statistical block on the winning model ({BEST_MODEL}), full test trajectory...")
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
residual_df.to_csv(config.OUTPUT_DIR / "residuals_full_trajectory.csv", index=False)

# Heteroscedasticity/autocorrelation tests are restricted to
# rul_true_traj <= RUL_CAP: the model was trained on the CAPPED target, so
# for cycles with true RUL > 125 the prediction is mechanically capped at
# ~125 while the "residual" keeps growing with the unbounded true RUL --
# that is not model heteroscedasticity, it's the arithmetic artifact of
# comparing a capped prediction against an uncapped target. The 11 censored
# engines are also excluded (their full trajectory lies above the cap).
# Without this filter, the cap itself generated an almost perfectly linear
# trend in the uncapped segment that dominated and artificially inflated the
# Breusch-Pagan statistic.
residual_capped = residual_df[residual_df["rul_true_traj"] <= config.RUL_CAP].copy()
log(f"  Filtering to rul_true_traj<={config.RUL_CAP} for the residual tests: "
    f"{len(residual_capped)}/{len(residual_df)} cycles, "
    f"{residual_capped['engine_id'].nunique()}/{residual_df['engine_id'].nunique()} engines "
    "(excludes the segment where the training target was capped, and the censored engines)")

bp = stats_eval.heteroscedasticity_by_engine(residual_capped)
log(f"  Real Breusch-Pagan (aggregated by engine, n={bp['n_engines']}, RUL<=cap only): "
    f"LM={bp['lm_stat']:.2f}, p={bp['lm_pvalue']:.2e}")

lb_df = stats_eval.ljung_box_by_engine(residual_capped, lags=10)
frac_autocorr = (lb_df["lb_pvalue"] < 0.05).mean()
log(f"  Ljung-Box (residuals centered per engine, lag=10, RUL<=cap only): "
    f"{frac_autocorr*100:.0f}% of engines show significant autocorrelation")

with open(config.OUTPUT_DIR / "residual_statistics.json", "w") as f:
    json.dump({
        "breusch_pagan_by_engine": bp,
        "ljung_box_frac_engines_autocorrelated": float(frac_autocorr),
        "note": "Tests computed only on cycles with rul_true_traj<=RUL_CAP "
                "(excludes the capped segment and the censored engines); see comment in train.py",
        "n_cycles_used": int(len(residual_capped)),
        "n_cycles_total": int(len(residual_df)),
        "n_engines_used": int(residual_capped["engine_id"].nunique()),
    }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 12. SINGLE RESULTS TABLE (source of truth for the README)
# ═══════════════════════════════════════════════════════════════════════════
log("Saving the single results table...")
rows = []
for name in modeling.MODEL_NAMES:
    rows.append({
        "model": name,
        "cv_rmse_mean": oof_fold_rmse[name].mean(), "cv_rmse_std": oof_fold_rmse[name].std(),
        "test_rmse": test_results[name]["rmse"], "test_mae": test_results[name]["mae"],
        "test_r2": test_results[name]["r2"], "test_phm": test_results[name]["phm"],
        "test_rmse_uncensored": test_results[name]["rmse_uncensored_lower_bound"],
    })
for name in baselines:
    rows.append({
        "model": name, "cv_rmse_mean": np.nan, "cv_rmse_std": np.nan,
        "test_rmse": test_results[name]["rmse"], "test_mae": test_results[name]["mae"],
        "test_r2": test_results[name]["r2"], "test_phm": test_results[name]["phm"],
        "test_rmse_uncensored": test_results[name]["rmse_uncensored_lower_bound"],
    })
results_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
results_df.to_csv(config.OUTPUT_DIR / "final_results.csv", index=False)
print(results_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# 13. FIGURES
# ═══════════════════════════════════════════════════════════════════════════
log("Generating figures...")


def savefig(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"  saved {path.name}")


# f01: sensor variance
fig, ax = plt.subplots(figsize=(10, 6))
stds = train_df[data.SENSOR_COLS].std().sort_values()
colors = [config.PALETTE["danger"] if s in low_var else config.PALETTE["primary"] for s in stds.index]
ax.barh(stds.index, stds.values, color=colors)
ax.axvline(config.VARIANCE_THRESHOLD, color=config.PALETTE["danger"], linestyle="--", label="Threshold")
ax.set_xlabel("Standard deviation (sensor already scaled to [0,1] globally)")
ax.set_title("Sensor variance -- FD001 train\nRed = dropped (std < threshold)")
ax.legend()
savefig(fig, "f01_sensor_variance.png")

# f02: raw vs capped RUL
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(train_rul["RUL_raw"], bins=50, color=config.PALETTE["neutral"])
axes[0].set_title("Raw RUL (uncapped)")
axes[1].hist(train_rul["RUL"], bins=50, color=config.PALETTE["primary"])
axes[1].set_title(f"RUL capped at {config.RUL_CAP} cycles")
for ax in axes:
    ax.set_xlabel("RUL (cycles)")
fig.suptitle("Effect of the target's piecewise-linear cap (Heimes 2008)")
savefig(fig, "f02_rul_distribution.png")

# f03: window ablation
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(window_df["windows"], window_df["cv_rmse_mean"], yerr=window_df["cv_rmse_std"],
       color=config.PALETTE["primary"], capsize=4)
ax.axhline(chosen_row["cv_rmse_mean"], color=config.PALETTE["secondary"], linestyle="--",
           label=f"Chosen: {WINDOWS}")
ax.set_ylabel("CV RMSE (cycles)")
ax.set_title("Temporal window ablation -- GroupKFold CV (LightGBM probe model)")
ax.tick_params(axis="x", rotation=20)
ax.legend()
savefig(fig, "f03_window_ablation.png")

# f04: s14 ablation
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["With s14", "Without s14"]
means = [scores_with.mean(), scores_without.mean()]
stds_ = [scores_with.std(), scores_without.std()]
ax.bar(labels, means, yerr=stds_, color=[config.PALETTE["primary"], config.PALETTE["accent"]], capsize=5)
ax.set_ylabel("CV RMSE (cycles)")
ax.set_title(f"s14 ablation (same pipeline, GroupKFold CV)\n"
             f"t={t_s14:.2f}, p={p_s14:.3f} -> {'keep' if keep_s14 else 'drop'}")
savefig(fig, "f04_s14_ablation.png")

# f05: CV comparison between models
fig, ax = plt.subplots(figsize=(9, 5))
names_sorted = cv_ranking
bp_data = [oof_fold_rmse[n] for n in names_sorted]
box = ax.boxplot(bp_data, tick_labels=names_sorted, patch_artist=True)
for patch, n in zip(box["boxes"], names_sorted):
    patch.set_facecolor(config.MODEL_COLORS[n])
    patch.set_alpha(0.7)
ax.set_ylabel("RMSE per fold (cycles)")
ax.set_title(f"Model comparison -- out-of-fold GroupKFold CV (k={config.N_OUTER_FOLDS})\n"
             f"Winner by CV: {BEST_MODEL}")
savefig(fig, "f05_cv_comparison.png")

# f06: test set -- RMSE/PHM bars + predicted vs actual for the winner
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
order = results_df["model"].tolist()
axes[0].bar(order, results_df["test_rmse"], color=config.PALETTE["primary"])
axes[0].set_title("RMSE -- official test set (100 engines)")
axes[0].tick_params(axis="x", rotation=60)
axes[1].bar(order, results_df["test_phm"], color=config.PALETTE["accent"])
axes[1].set_title("PHM score (Saxena 2008, lower=better)")
axes[1].tick_params(axis="x", rotation=60)
# Log scale: on a linear scale the baselines (PHM in the thousands/hundreds
# of thousands) would flatten the 4 real models' bars (PHM in the hundreds)
# into something visually indistinguishable from 0 -- exactly what made this
# figure illegible before.
axes[1].set_yscale("log")
axes[1].set_ylabel("PHM score (log scale)")
best_preds = test_results[BEST_MODEL]["preds"]
order_idx = np.argsort(y_test)
axes[2].scatter(range(len(y_test)), y_test[order_idx], color=config.PALETTE["neutral"], label="True RUL", s=25)
axes[2].scatter(range(len(y_test)), best_preds[order_idx], color=config.PALETTE["primary"],
                label=f"{BEST_MODEL} (predicted)", s=25, alpha=0.8)
axes[2].fill_between(range(len(y_test)), y_test[order_idx] - 20, y_test[order_idx] + 20,
                      color=config.PALETTE["neutral"], alpha=0.12, label="+/-20 cycles")
axes[2].set_title(f"Predicted vs. actual -- {BEST_MODEL}")
axes[2].legend(fontsize=8)
fig.suptitle("Single evaluation on the official test set (touched once)")
savefig(fig, "f06_test_evaluation.png")

# f07: residuals vs true RUL (heteroscedasticity). The full trajectory is
# shown for context, but the RUL>cap segment is grayed out (where the
# prediction is mechanically capped at ~125 while the x-axis is not -- that
# segment does NOT enter the Breusch-Pagan test, which is computed only on
# RUL<=cap; showing it undistinguished would make model heteroscedasticity
# out of what is largely an artifact of the cap).
fig, ax = plt.subplots(figsize=(8, 5))
sample = residual_df.sample(min(5000, len(residual_df)), random_state=config.SEED)
above_cap = sample["rul_true_traj"] > config.RUL_CAP
ax.scatter(sample.loc[above_cap, "rul_true_traj"], sample.loc[above_cap, "residual"],
           s=6, alpha=0.15, color=config.PALETTE["neutral"],
           label=f"RUL>{config.RUL_CAP} (outside the training target, excluded from the test)")
ax.scatter(sample.loc[~above_cap, "rul_true_traj"], sample.loc[~above_cap, "residual"],
           s=6, alpha=0.3, color=config.PALETTE["primary"], label=f"RUL<={config.RUL_CAP} (used in the test)")
ax.axvline(config.RUL_CAP, color=config.PALETTE["danger"], linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(0, color="black", linewidth=1)
ax.invert_xaxis()
ax.set_xlabel("True RUL (remaining cycles)")
ax.set_ylabel("Residual (true RUL - predicted RUL)")
ax.legend(fontsize=7, loc="upper left")
ax.set_title(f"Residuals per cycle -- {BEST_MODEL}, full test trajectory\n"
             f"Breusch-Pagan (aggregated by engine, n={bp['n_engines']}, RUL<={config.RUL_CAP} only): "
             f"p={bp['lm_pvalue']:.1e}")
savefig(fig, "f07_residuals_heteroscedasticity.png")

# f08: SHAP summary (global) for the winning model -- on the test set already
# touched, for interpretability only, it decides nothing in the pipeline.
try:
    inner_model = best_final.named_steps["model"]
    explainer = shap.TreeExplainer(inner_model)
    shap_values = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, feature_names=FINAL_COLS, show=False, max_display=15)
    plt.title(f"SHAP summary -- {BEST_MODEL} (interpretation, does not decide the pipeline)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "f08_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    log("  saved f08_shap_summary.png")
except Exception as exc:  # pragma: no cover
    log(f"  [warning] SHAP not available for {BEST_MODEL}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# 14. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
summary = {
    "informative_sensors": informative,
    "sensors_dropped_by_variance": low_var,
    "windows_chosen_by_cv": WINDOWS,
    "s14_kept_by_cv": bool(keep_s14),
    "n_features_final": len(FINAL_COLS),
    "winning_model_by_cv": BEST_MODEL,
    "censored_test_engines": n_censored,
    "total_time_seconds": round(time.time() - T0, 1),
}
with open(config.OUTPUT_DIR / "pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

log("=" * 70)
log("PIPELINE COMPLETE")
for k, v in summary.items():
    log(f"  {k}: {v}")
log("=" * 70)
