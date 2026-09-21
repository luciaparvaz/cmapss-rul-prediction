"""Evaluación y contrastes estadísticos correctos entre modelos.

Corrige los 6 defectos que señaló la auditoría (Bloque 6):
  - "Diebold-Mariano" -> es un t-test pareado sobre 100 motores (dimensión
    transversal, no temporal); se renombra y se elimina la corrección HLN,
    que estaba aplicada al revés (dividía en vez de multiplicar, y el
    término h(h-1)/n con h=1 es 0, no 1/n).
  - Comparaciones múltiples sin corrección -> Holm sobre las 6 comparaciones
    posibles entre los 4 modelos.
  - "Breusch-Pagan" -> en realidad Glejser con gl inflados x132 (13096
    residuos de solo 100 motores, ignorando el clustering). Aquí se agrega a
    nivel de motor (100 observaciones) antes de aplicar el BP.
  - "Durbin-Watson" sobre residuos no centrados y sin distribución nula
    válida para un ensemble -> se sustituye por Ljung-Box sobre residuos
    centrados por motor.
  - Off-by-one en la reconstrucción de la trayectoria de RUL real.
  - IC bootstrap pareado (por motor) para las diferencias de RMSE/PHM.

Implementado únicamente con numpy/scipy.stats (sin statsmodels: en este
entorno statsmodels 0.14.4 es incompatible con scipy 1.17 -- ver nota en
requirements.txt -- así que Breusch-Pagan, Ljung-Box y Holm se calculan a
mano con las fórmulas estándar, sin cambiar el resultado estadístico).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, ttest_rel


def phm_score(y_true: np.ndarray, y_pred: np.ndarray, a_early: float = 13.0, a_late: float = 10.0) -> np.ndarray:
    """PHM score asimétrico de Saxena et al. (2008), por motor (menor = mejor).

    d = pred - true. Penaliza más la sobreestimación (d>0, predecir más vida
    de la que realmente queda: retrasa el mantenimiento, dirección peligrosa
    en un motor de avión) con a_late; la subestimación con a_early. Los
    valores estándar de Saxena son a1=13 (early, d<0) y a2=10 (late, d>=0):
    a2 < a1 hace que exp(d/a2) crezca más rápido que exp(-d/a1) para el mismo
    |d|, de modo que la sobreestimación queda penalizada con más fuerza. (Una
    versión anterior de este archivo tenía a_early=10/a_late=13 -- los
    valores intercambiados -- lo que invertía esa asimetría sin que ningún
    test lo detectara; ver commit de corrección.)
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / a_early) - 1, np.exp(d / a_late) - 1)
    return s


def paired_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """t-test pareado sobre la pérdida cuadrática por motor. NO es un test de
    Diebold-Mariano (ese exige una única serie temporal con horizonte h y
    varianza de largo plazo HAC; aquí la unidad de pareo son 100 motores
    distintos, una comparación transversal)."""
    t, p = ttest_rel(errors_a, errors_b)
    return float(t), float(p)


def holm_correction(pvalues: dict[str, float]) -> pd.DataFrame:
    """Corrección de Holm-Bonferroni (step-down), implementación manual
    estándar (Holm 1979): ordena p-valores ascendente, compara p_(i) con
    alfa/(m-i+1); una vez no se rechaza, no se rechaza ninguno posterior."""
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
        "comparacion": names,
        "p_valor": pvals,
        "p_holm": p_adj,
        "significativo_holm_0.05": reject,
    })


def bootstrap_ci_diff_by_engine(errors_a: np.ndarray, errors_b: np.ndarray,
                                 n_boot: int = 5000, seed: int = 42,
                                 alpha: float = 0.05) -> dict:
    """IC bootstrap (percentil) de la diferencia media de error pareada por
    motor, remuestreando motores con reemplazo."""
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
    """RUL real de cada ciclo observado de un motor de test, SIN el
    off-by-one del proyecto original (`arange(n_cyc+rul_true, rul_true, -1)`
    terminaba en rul_true+1, no en rul_true). Devuelve un array de longitud
    n_cycles donde el último valor es exactamente rul_true."""
    start = n_cycles - 1 + rul_true
    stop = rul_true - 1
    return np.arange(start, stop, -1)


def _breusch_pagan_manual(y: np.ndarray, x: np.ndarray) -> dict:
    """Breusch-Pagan sobre una regresión y ~ const + x (una sola variable).

    LM = n * R² de la regresión auxiliar, ~ chi2(k) bajo H0 de homocedasticidad
    (k = nº de regresores sin la constante). También se reporta la versión F.
    Fórmulas estándar (Breusch & Pagan 1979); implementación directa con
    numpy.linalg.lstsq en vez de statsmodels (incompatible en este entorno).
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
    """Breusch-Pagan sobre datos agregados por motor (n=100, no n=13096):
    para cada motor se toma el error cuadrático medio de su trayectoria y el
    RUL medio observado, y se contrasta si la varianza del residuo depende
    del nivel de RUL. Evita inflar los grados de libertad por
    pseudo-replicación intra-motor (13096 ciclos de solo 100 motores).

    residual_df: columnas engine_id, rul_true_traj, residual
    """
    agg = residual_df.groupby("engine_id").agg(
        mean_sq_resid=("residual", lambda s: float(np.mean(s.to_numpy() ** 2))),
        mean_rul=("rul_true_traj", "mean"),
    ).reset_index()

    bp = _breusch_pagan_manual(agg["mean_sq_resid"].to_numpy(), agg["mean_rul"].to_numpy())
    bp["n_motores"] = len(agg)
    return bp


def _autocorr(x: np.ndarray, lag: int) -> float:
    n = len(x)
    x = x - x.mean()
    num = np.sum(x[: n - lag] * x[lag:])
    den = np.sum(x ** 2)
    return float(num / den) if den > 0 else 0.0


def _ljung_box_manual(residuals: np.ndarray, lags: int) -> tuple[float, float]:
    """Estadístico de Ljung-Box (1978): LB = n(n+2) * sum_{k=1..h} r_k^2/(n-k),
    ~ chi2(h) bajo H0 de no autocorrelación. Implementación directa (sin
    statsmodels, incompatible en este entorno)."""
    n = len(residuals)
    lb = 0.0
    for k in range(1, lags + 1):
        r_k = _autocorr(residuals, k)
        lb += (r_k ** 2) / (n - k)
    lb *= n * (n + 2)
    p = float(chi2.sf(lb, df=lags))
    return float(lb), p


def ljung_box_by_engine(residual_df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box sobre residuos CENTRADOS por motor (resta la media del
    propio motor antes de testear autocorrelación; el sesgo medio por motor
    inflaba artificialmente la autocorrelación aparente en el DW original).
    Se ejecuta un test por motor y se reporta la fracción de motores con
    autocorrelación significativa."""
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
