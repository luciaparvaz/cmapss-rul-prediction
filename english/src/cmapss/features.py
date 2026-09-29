"""Feature engineering -- the SINGLE definition used by the whole pipeline.

There used to be three incompatible definitions (phase2: w={15,30,50}/5
stats; phase5/phase7: w={15,30}/3 stats; the saved .pkl files trained with a
third one). This is the single source of truth; the final window size is
decided by cross-validation in the ablation phase of train.py, not by hand.

All features are strictly causal (trailing): at cycle t, only information
from cycles <= t of the same engine is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ("mean", "std", "slope")


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope over a trailing rolling window, vectorized via convolution.

    slope(t) is computed over cycles [max(0, t-window+1), t]. For t with
    fewer than `window` observations available, a partial window is used
    (never looks ahead).
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

    # partial windows for the first w-1 cycles
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
    """Adds, per engine and sensor: {s}_w{w}_mean/std/slope for each w, and
    {s}_delta = current_value - value_at_engine's_first_cycle (cumulative
    drift, uses only the first observed cycle, never future information).

    All new columns are accumulated in a dict and concatenated once at the
    end (instead of assigning column by column onto the DataFrame, which
    with >100 new columns fragments pandas' memory block and degrades
    performance quadratically).
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
    """Column names of the feature vector, in the order `add_rolling_features`
    generates them + the raw sensors (already globally scaled). This is the
    real whitelist: nothing else enters the model (fixes the original
    phase-4 bug, which included zero-variance, un-normalized sensors because
    it took "anything that isn't a target")."""
    cols = list(sensors)
    for s in sensors:
        for w in windows:
            cols += [f"{s}_w{w}_mean", f"{s}_w{w}_std", f"{s}_w{w}_slope"]
    for s in sensors:
        cols.append(f"{s}_delta")
    return cols


def build_features(df: pd.DataFrame, sensors: list[str], windows: list[int]) -> pd.DataFrame:
    """Orchestrates: rolling features + delta. `df` must already carry the
    globally-scaled sensors and the RUL target."""
    return add_rolling_features(df, sensors, windows)
