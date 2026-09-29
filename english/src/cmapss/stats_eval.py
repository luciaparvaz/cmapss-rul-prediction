"""Correct statistical evaluation and model comparison.

Fixes 6 defects in the original project:
  - "Diebold-Mariano" -> is actually a paired t-test over 100 engines (a
    cross-sectional comparison, not a temporal one); renamed, and the HLN
    correction is removed (it was applied backwards -- dividing instead of
    multiplying -- and the h(h-1)/n term with h=1 is 0, not 1/n).
  - Multiple comparisons without correction -> Holm over the 6 possible
    comparisons between the 4 models.
  - "Breusch-Pagan" -> actually a Glejser test with degrees of freedom
    inflated x132 (13,096 residuals from only 100 engines, ignoring
    clustering). Here it is aggregated at the engine level (100
    observations) before applying BP.
  - "Durbin-Watson" on non-centered residuals with no valid null
    distribution for an ensemble -> replaced with Ljung-Box on residuals
    centered per engine.
  - Off-by-one in the reconstruction of the true RUL trajectory.
  - Paired bootstrap CI (per engine) for RMSE/PHM differences.

Implemented using only numpy/scipy.stats (no statsmodels: in this
environment statsmodels 0.14.4 is incompatible with scipy 1.17 -- see the
note in requirements.txt -- so Breusch-Pagan, Ljung-Box and Holm are computed
by hand with the standard formulas, without changing the statistical
result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, ttest_rel


def phm_score(y_true: np.ndarray, y_pred: np.ndarray, a_early: float = 13.0, a_late: float = 10.0) -> np.ndarray:
    """Asymmetric PHM score from Saxena et al. (2008), per engine (lower = better).

    d = pred - true. Penalizes overestimation more heavily (d>0, predicting
    more remaining life than there actually is: delays maintenance, the
    dangerous direction in an aircraft engine) via a_late; underestimation is
    penalized via a_early. Saxena's standard values are a1=13 (early, d<0)
    and a2=10 (late, d>=0): a2 < a1 makes exp(d/a2) grow faster than
    exp(-d/a1) for the same |d|, so overestimation ends up penalized more
    strongly. (An earlier version of this file had a_early=10/a_late=13 --
    the values swapped -- which inverted that asymmetry without any test
    catching it; see the fix commit.)
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / a_early) - 1, np.exp(d / a_late) - 1)
    return s


def paired_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """Paired t-test on the squared loss per engine. This is NOT a
    Diebold-Mariano test (that requires a single time series with a horizon h
    and a long-run HAC variance; here the pairing unit is 100 distinct
    engines, a cross-sectional comparison)."""
    t, p = ttest_rel(errors_a, errors_b)
    return float(t), float(p)


def holm_correction(pvalues: dict[str, float]) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction, standard manual implementation
    (Holm 1979): sorts p-values ascending, compares p_(i) against
    alpha/(m-i+1); once one fails to reject, none after it is rejected
    either."""
    names = list(pvalues.keys())
    pvals = np.array([pvalues[n] for n in names])
    m = len(pvals)
    order = np.argsort(pvals)
    p_adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        p_adj[idx] = min(running_max, 1.0)
    reject = p_adj < 0.05
    return pd.DataFrame({
        "comparison": names,
        "p_value": pvals,
        "p_holm": p_adj,
        "significant_holm_0.05": reject,
    })


def bootstrap_ci_diff_by_engine(errors_a: np.ndarray, errors_b: np.ndarray,
                                 n_boot: int = 5000, seed: int = 42,
                                 alpha: float = 0.05) -> dict:
    """Bootstrap (percentile) CI of the mean paired error difference per
    engine, resampling engines with replacement."""
    rng = np.random.default_rng(seed)
    diff = errors_a - errors_b
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff_mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def reconstruct_true_rul_trajectory(n_cycles: int, rul_true: float) -> np.ndarray:
    """True RUL for each observed cycle of a test engine, WITHOUT the
    off-by-one from the original project (`arange(n_cyc+rul_true, rul_true,
    -1)` used to end at rul_true+1, not rul_true). Returns an array of length
    n_cycles whose last value is exactly rul_true."""
    start = n_cycles - 1 + rul_true
    stop = rul_true - 1
    return np.arange(start, stop, -1)


def _breusch_pagan_manual(y: np.ndarray, x: np.ndarray) -> dict:
    """Breusch-Pagan on a regression y ~ const + x (a single variable).

    LM = n * R² of the auxiliary regression, ~ chi2(k) under H0 of
    homoscedasticity (k = number of regressors excluding the constant). The
    F version is also reported. Standard formulas (Breusch & Pagan 1979);
    implemented directly with numpy.linalg.lstsq instead of statsmodels
    (incompatible in this environment).
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = X.shape[1] - 1

    lm_stat = n * r2
    lm_pvalue = float(chi2.sf(lm_stat, df=k))

    if r2 < 1.0 and n - k - 1 > 0:
        f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
        f_pvalue = float(f_dist.sf(f_stat, k, n - k - 1))
    else:
        f_stat, f_pvalue = float("nan"), float("nan")

    return {"lm_stat": lm_stat, "lm_pvalue": lm_pvalue, "f_stat": f_stat, "f_pvalue": f_pvalue}


def heteroscedasticity_by_engine(residual_df: pd.DataFrame) -> dict:
    """Breusch-Pagan on data aggregated by engine (n=100, not n=13,096): for
    each engine, take the mean squared error of its trajectory and its mean
    observed RUL, and test whether the residual variance depends on the RUL
    level. Avoids inflating degrees of freedom via intra-engine
    pseudo-replication (13,096 cycles from only 100 engines).

    residual_df: columns engine_id, rul_true_traj, residual
    """
    agg = residual_df.groupby("engine_id").agg(
        mean_sq_resid=("residual", lambda s: float(np.mean(s.to_numpy() ** 2))),
        mean_rul=("rul_true_traj", "mean"),
    ).reset_index()

    bp = _breusch_pagan_manual(agg["mean_sq_resid"].to_numpy(), agg["mean_rul"].to_numpy())
    bp["n_engines"] = len(agg)
    return bp


def _autocorr(x: np.ndarray, lag: int) -> float:
    n = len(x)
    x = x - x.mean()
    num = np.sum(x[: n - lag] * x[lag:])
    den = np.sum(x ** 2)
    return float(num / den) if den > 0 else 0.0


def _ljung_box_manual(residuals: np.ndarray, lags: int) -> tuple[float, float]:
    """Ljung-Box statistic (1978): LB = n(n+2) * sum_{k=1..h} r_k^2/(n-k),
    ~ chi2(h) under H0 of no autocorrelation. Implemented directly (no
    statsmodels, incompatible in this environment)."""
    n = len(residuals)
    lb = 0.0
    for k in range(1, lags + 1):
        r_k = _autocorr(residuals, k)
        lb += (r_k ** 2) / (n - k)
    lb *= n * (n + 2)
    p = float(chi2.sf(lb, df=lags))
    return float(lb), p


def ljung_box_by_engine(residual_df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box on residuals CENTERED per engine (subtracts each engine's own
    mean before testing autocorrelation; the per-engine mean bias artificially
    inflated the apparent autocorrelation in the original DW). Runs one test
    per engine and reports the fraction of engines with significant
    autocorrelation."""
    results = []
    for engine_id, g in residual_df.groupby("engine_id"):
        resid = g.sort_values("cycle")["residual"].to_numpy()
        resid = resid - resid.mean()
        if len(resid) <= lags + 1:
            continue
        lb_stat, lb_pvalue = _ljung_box_manual(resid, lags)
        results.append({
            "engine_id": engine_id,
            "n_cycles": len(resid),
            "lb_stat": lb_stat,
            "lb_pvalue": lb_pvalue,
        })
    return pd.DataFrame(results)
