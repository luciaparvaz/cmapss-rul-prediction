"""Ingeniería de features -- ÚNICA definición usada por todo el pipeline.

Antes existían tres definiciones incompatibles (fase2: w={15,30,50}/5 stats;
fase5/fase7: w={15,30}/3 stats; los .pkl entrenados con una tercera). Esta es
la única fuente de verdad; el tamaño de ventana final se decide por
validación cruzada en `ablation.select_windows`, no a mano.

Todas las features son estrictamente causales (trailing): en el ciclo t solo
se usa información de ciclos <= t del mismo motor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ("mean", "std", "slope")


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Pendiente OLS en ventana deslizante trailing, vectorizada por convolución.

    slope(t) se calcula sobre los ciclos [max(0, t-window+1), t]. Para t con
    menos de `window` observaciones disponibles se usa una ventana parcial
    (nunca se mira hacia adelante).
    """
    n = len(values)
    slopes = np.zeros(n, dtype=np.float64)
    if n == 0:
        return slopes

    w = window
    x_full = np.arange(w, dtype=np.float64)
    x_full -= x_full.mean()
    denom = (x_full ** 2).sum()

    if denom > 0 and n >= w:
        conv = np.convolve(values.astype(np.float64), x_full[::-1], mode="full")
        slopes[w - 1:] = conv[w - 1:n] / denom

    # ventanas parciales para los primeros w-1 ciclos
    upper = min(w - 1, n)
    for t in range(1, upper):
        y_win = values[: t + 1]
        x_win = np.arange(t + 1, dtype=np.float64)
        x_win -= x_win.mean()
        v = (x_win ** 2).sum()
        if v > 0:
            slopes[t] = np.dot(x_win, y_win) / v
    return slopes


def add_rolling_features(df: pd.DataFrame, sensors: list[str], windows: list[int]) -> pd.DataFrame:
    """Añade, por motor y sensor: {s}_w{w}_mean/std/slope para cada w, y
    {s}_delta = valor_actual - valor_primer_ciclo_del_motor (deriva acumulada,
    usa solo el primer ciclo observado, nunca información futura).

    Todas las columnas nuevas se acumulan en un dict y se concatenan una
    sola vez al final (en vez de asignar columna a columna sobre el
    DataFrame, que con >100 columnas nuevas fragmenta el bloque de memoria
    de pandas y degrada el rendimiento cuadráticamente).
    """
    n = len(df)
    new_cols: dict[str, np.ndarray] = {
        name: np.zeros(n, dtype=np.float64)
        for col in sensors
        for name in ([f"{col}_delta"] + [f"{col}_w{w}_{stat}" for w in windows for stat in STATS])
    }

    for engine_id, group in df.groupby("engine_id", sort=False):
        pos = df.index.get_indexer(group.index)
        for col in sensors:
            v = group[col].to_numpy(dtype=np.float64)
            new_cols[f"{col}_delta"][pos] = v - v[0]
            for w in windows:
                ser = pd.Series(v)
                new_cols[f"{col}_w{w}_mean"][pos] = ser.rolling(w, min_periods=1).mean().to_numpy()
                new_cols[f"{col}_w{w}_std"][pos] = ser.rolling(w, min_periods=1).std(ddof=1).fillna(0.0).to_numpy()
                new_cols[f"{col}_w{w}_slope"][pos] = _rolling_slope(v, w)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def feature_columns(sensors: list[str], windows: list[int]) -> list[str]:
    """Nombres de columnas del vector de features, en el orden en que las
    genera `add_rolling_features` + los sensores crudos (ya escalados
    globalmente). Es la lista blanca real: nada más entra al modelo
    (corrige el bug de fase4, que incluía sensores de varianza nula sin
    normalizar porque tomaba "todo lo que no es meta")."""
    cols = list(sensors)
    for s in sensors:
        for w in windows:
            cols += [f"{s}_w{w}_mean", f"{s}_w{w}_std", f"{s}_w{w}_slope"]
    for s in sensors:
        cols.append(f"{s}_delta")
    return cols


def build_features(df: pd.DataFrame, sensors: list[str], windows: list[int]) -> pd.DataFrame:
    """Orquesta: rolling features + delta. `df` debe traer ya los sensores
    escalados globalmente (ver pipeline.scale_sensors) y el target RUL."""
    return add_rolling_features(df, sensors, windows)
