"""Pipelines, hyperparameter search spaces, and baselines.

All scaling lives INSIDE a sklearn.Pipeline, so it is never fit outside the
CV loop (the StandardScaler used to be fit outside CV and, worse, the saved
final scaler didn't match the set actually used for training -> RMSE ~19.8
instead of ~12.8 when reloaded).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from . import config

RUL_CAP = config.RUL_CAP


def build_pipeline(name: str) -> Pipeline:
    """Ridge needs scaling; tree models are invariant to monotonic per-feature
    transforms, so they get passthrough (avoids scaling unnecessarily, but the
    Pipeline has the same shape for all 4 models: a single source of truth,
    never a loose scaler)."""
    if name == "Ridge":
        scaler = StandardScaler()
        model = Ridge(random_state=None)
    elif name == "RandomForest":
        # RandomForest (exact split, not histogram-based) is much slower than
        # XGBoost/LightGBM on this dataset; it parallelizes INSIDE the model
        # itself (n_jobs=-1), and the search/CV wrapping it runs in series
        # (see SEARCH_NJOBS/CV_NJOBS in train.py) to avoid over-subscribing
        # the cores with two levels of parallelism at once.
        scaler = "passthrough"
        model = RandomForestRegressor(random_state=config.SEED, n_jobs=-1)
    elif name == "XGBoost":
        scaler = "passthrough"
        model = xgb.XGBRegressor(
            random_state=config.SEED, tree_method="hist", device="cpu", verbosity=0, n_jobs=1,
        )
    elif name == "LightGBM":
        scaler = "passthrough"
        model = lgb.LGBMRegressor(random_state=config.SEED, verbose=-1, n_jobs=1)
    else:
        raise ValueError(name)
    return Pipeline([("scaler", scaler), ("model", model)])


PARAM_DISTRIBUTIONS = {
    "Ridge": {"model__alpha": uniform(0.1, 100)},
    "RandomForest": {
        # Range trimmed on purpose: RandomForestRegressor (exact split, not
        # histogram-based) is ~10-20x slower than XGBoost/LightGBM on this
        # dataset (see performance note in train.py); above ~220
        # n_estimators the hyperparameter search becomes impractical on CPU.
        # n_iter is also reduced for this model (see SEARCH_ITER_OVERRIDE).
        "model__n_estimators": randint(80, 220),
        "model__max_depth": randint(8, 18),
        "model__min_samples_leaf": randint(2, 10),
        "model__max_features": uniform(0.2, 0.6),
    },
    "XGBoost": {
        "model__n_estimators": randint(150, 500),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": uniform(0.02, 0.13),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": uniform(0.0, 0.5),
        "model__reg_lambda": uniform(0.5, 2.0),
    },
    "LightGBM": {
        "model__n_estimators": randint(150, 500),
        "model__num_leaves": randint(15, 90),
        "model__learning_rate": uniform(0.02, 0.13),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": uniform(0.0, 0.5),
        "model__reg_lambda": uniform(0.5, 2.0),
        "model__min_child_samples": randint(10, 40),
    },
}

MODEL_NAMES = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]


# RandomForest is much slower to fit (see note above); compensated with
# fewer search combinations to keep the total runtime reasonable.
SEARCH_ITER_OVERRIDE = {"Ridge": 10, "RandomForest": 8}


def search_iter(name: str, default: int) -> int:
    return SEARCH_ITER_OVERRIDE.get(name, default)


def outer_njobs(name: str) -> int:
    """n_jobs for the RandomizedSearchCV/cross_val_predict calls that WRAP the
    model. RandomForest already parallelizes internally (n_jobs=-1 on the
    estimator itself); wrapping it with another layer of parallelism
    oversubscribes the cores and in practice SLOWS IT DOWN. The other models
    are single-threaded internally, so the parallelism lives in the outer
    layer instead."""
    return 1 if name == "RandomForest" else -1


def clip_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(preds, 0.0, RUL_CAP)


# ---------------------------------------------------------------------------
# Trivial baselines (absent from the original project)
# ---------------------------------------------------------------------------

def baseline_predictions(y_train: np.ndarray, train_mean_life: float,
                          test_cycles_observed: np.ndarray, n_test: int) -> dict[str, np.ndarray]:
    """Three baselines that require no features at all:
      - train mean
      - train median
      - train mean life minus cycles already observed (clipped to [0, RUL_CAP])
    """
    mean_pred = np.full(n_test, y_train.mean())
    median_pred = np.full(n_test, np.median(y_train))
    life_minus_observed = np.clip(train_mean_life - test_cycles_observed, 0, RUL_CAP)
    return {
        "Baseline-Mean": clip_predictions(mean_pred),
        "Baseline-Median": clip_predictions(median_pred),
        "Baseline-MeanLife-Cycles": clip_predictions(life_minus_observed),
    }
