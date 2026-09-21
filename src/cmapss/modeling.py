"""Pipelines, espacios de búsqueda de hiperparámetros y baselines.

Todo escalado vive DENTRO de un sklearn.Pipeline, así que nunca se ajusta
fuera del bucle de CV (corrige el hallazgo 2.6/4.4 de la auditoría: el
StandardScaler se ajustaba fuera del CV y, peor, el scaler final guardado no
correspondía al conjunto realmente usado para entrenar -> RMSE ~19.8 en vez
de ~12.8 al recargar).
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
    """Ridge necesita escalado; los modelos de árboles son invariantes a
    transformaciones monótonas por feature, así que se deja passthrough
    (evita escalar sin necesidad, pero el Pipeline es idéntico en forma
    para los 4 modelos: una sola fuente de verdad, nunca un scaler suelto)."""
    if name == "Ridge":
        scaler = StandardScaler()
        model = Ridge(random_state=None)
    elif name == "RandomForest":
        # RandomForest (split exacto, no histograma) es mucho más lento que
        # XGBoost/LightGBM en este dataset; se paraleliza DENTRO del propio
        # modelo (n_jobs=-1) y la búsqueda/CV que lo envuelve se ejecuta en
        # serie (ver SEARCH_NJOBS/CV_NJOBS en train.py) para no sobre-suscribir
        # los núcleos con dos niveles de paralelismo a la vez.
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
        # Rango recortado a propósito: RandomForestRegressor (split exacto,
        # no histograma) es ~10-20x más lento que XGBoost/LightGBM en este
        # dataset (ver nota de rendimiento en train.py); con n_estimators
        # por encima de ~220 la búsqueda de hiperparámetros se vuelve
        # impracticable en CPU. n_iter también se reduce para este modelo
        # (ver SEARCH_ITER_OVERRIDE).
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


# RandomForest es mucho más lento por fit (ver nota arriba); se compensa con
# menos combinaciones de búsqueda para mantener el tiempo total razonable.
SEARCH_ITER_OVERRIDE = {"Ridge": 10, "RandomForest": 8}


def search_iter(name: str, default: int) -> int:
    return SEARCH_ITER_OVERRIDE.get(name, default)


def outer_njobs(name: str) -> int:
    """n_jobs para RandomizedSearchCV/cross_val_predict que ENVUELVEN el
    modelo. RandomForest ya paraleliza internamente (n_jobs=-1 en el propio
    estimador); envolverlo con otro nivel de paralelismo satura los núcleos
    y en la práctica LO RALENTIZA. Los demás modelos son single-thread
    internamente, así que el paralelismo vive en la capa exterior."""
    return 1 if name == "RandomForest" else -1


def clip_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(preds, 0.0, RUL_CAP)


# ---------------------------------------------------------------------------
# Baselines triviales (ausentes en el proyecto original -- auditoría 5.6)
# ---------------------------------------------------------------------------

def baseline_predictions(y_train: np.ndarray, train_mean_life: float,
                          test_cycles_observed: np.ndarray, n_test: int) -> dict[str, np.ndarray]:
    """Tres baselines que no requieren ningún feature:
      - media del train
      - mediana del train
      - vida media del train menos ciclos ya observados (capeado a RUL_CAP y a 0)
    """
    mean_pred = np.full(n_test, y_train.mean())
    median_pred = np.full(n_test, np.median(y_train))
    life_minus_observed = np.clip(train_mean_life - test_cycles_observed, 0, RUL_CAP)
    return {
        "Baseline-Media": clip_predictions(mean_pred),
        "Baseline-Mediana": clip_predictions(median_pred),
        "Baseline-VidaMedia-Ciclos": clip_predictions(life_minus_observed),
    }
