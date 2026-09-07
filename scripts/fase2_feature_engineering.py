"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 2: Ingeniería de características (a partir de Fase 0)
=============================================================================
Esta versión sustituye a la Fase 2 original, que partía de datos
normalizados POR MOTOR (el bug de leakage corregido en Fase 0) y usaba
windows=[15,30,50] con 5 estadísticos (mean/std/min/max/slope) → 238
features, un pipeline nunca usado realmente para entrenar los modelos
guardados en models/.

Aquí se construyen y comparan EXPLÍCITAMENTE dos conjuntos de features
sobre los mismos datos correctamente normalizados (Fase 0):

  A. CANÓNICO (112 features) — windows=[15,30], mean/std/slope + delta
     acumulado. Es, feature a feature, el pipeline que reconstruían ad
     hoc fase5_evaluacion.py y fase7_mejoras.py para poder usar los
     modelos ya guardados (xgboost_fd001.pkl, etc.). Se usa en Fase 4.

  B. EXTENDIDO (238 features) — windows=[15,30,50], mean/std/min/max/
     slope + delta de primer orden. Es la propuesta de la Fase 2
     original. Se mantiene aquí solo como comparación/ablación, no
     alimenta Fase 4.

La comparación empírica (sección final) entrena un LightGBM ligero con
GroupKFold(k=5) sobre cada conjunto y compara RMSE, para justificar con
datos —no solo con el argumento de "menos features, menos sobreajuste"—
por qué Fase 4 usa el conjunto canónico. Esto es coherente con (y una
versión reducida de) la ablación de ventanas de fase7_mejoras.py (M6),
que evalúa 10 combinaciones de ventanas con LightGBM y GroupKFold y
concluye que w={15,30} es el mejor compromiso complejidad/error.

Referencias:
  - Heimes, F. O. (2008). Recurrent neural networks for remaining useful
    life estimation. PHM 2008. — RUL cap a 125 ciclos.
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
    Statistical Learning (2nd ed.). Springer, cap. 7 (selección de
    modelos vía validación cruzada; sesgo-varianza).
  - Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting
    decision tree. NeurIPS 30.
=============================================================================
"""

import os
import sys
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, cross_val_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fase0_preprocesado as f0

OUTPUT_DIR = f0.OUTPUT_DIR
MODEL_DIR  = f0.MODEL_DIR
FIG_DIR    = f0.FIG_DIR

RUL_CAP     = 125    # ciclos (Heimes 2008)
FAIL_THRESH = 30     # ciclos para early_failure

# Definición de los dos conjuntos de features a comparar
CANONICAL_WINDOWS = [15, 30]
CANONICAL_STATS    = ["mean", "std", "slope"]
EXTENDED_WINDOWS  = [15, 30, 50]
EXTENDED_STATS     = ["mean", "std", "min", "max", "slope"]


# ─────────────────────────────────────────────
# 1. CARGA DE SALIDAS DE FASE 0
# ─────────────────────────────────────────────

def load_fase0_outputs():
    """Carga los parquets normalizados globalmente (Fase 0) y la lista
    de sensores informativos usada para generarlos."""
    train_norm = pd.read_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet")
    test_norm  = pd.read_parquet(f"{OUTPUT_DIR}/test_FD001_normalized.parquet")
    rul_df     = pd.read_parquet(f"{OUTPUT_DIR}/RUL_FD001.parquet")
    informative = f0.INFORMATIVE_EXPECTED
    missing = [s for s in informative if s not in train_norm.columns]
    if missing:
        raise RuntimeError(
            f"Faltan columnas de sensores informativos en la salida de "
            f"Fase 0: {missing}. Ejecuta fase0_preprocesado.py primero."
        )
    return train_norm, test_norm, rul_df, informative


# ─────────────────────────────────────────────
# 2. TARGET RUL
# ─────────────────────────────────────────────

def compute_rul_train(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    RUL_raw = ciclo_máximo_del_motor − ciclo_actual.
    RUL = min(RUL_raw, rul_cap)  — piece-wise linear degradation
    (Heimes, 2008): en la primera parte de la vida del motor no hay
    señal de degradación medible, así que capar el target evita pedirle
    al modelo que prediga una cantidad indistinguible del ruido.
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL_raw"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)
    df["early_failure"] = (df["RUL_raw"] <= FAIL_THRESH).astype(int)
    return df


def compute_rul_test_full(test_df: pd.DataFrame, rul_df: pd.DataFrame,
                           rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    A diferencia de la Fase 2 original (que solo asignaba RUL al último
    ciclo observado de test, dejando el resto en NaN), aquí se calcula
    el RUL verdadero en TODOS los ciclos de la trayectoria de test por
    cuenta atrás desde el RUL final conocido:

        RUL(t) = (n_ciclos_restantes_hasta_truncar) + RUL_final − (cycle − 1)

    Esto es exactamente lo que fase7_mejoras.py (Mejora 5) reconstruye
    ad hoc para poder evaluar cada motor ciclo a ciclo; centralizarlo
    aquí evita reimplementarlo por tercera vez.
    """
    df = test_df.copy()
    df["RUL"] = np.nan
    for eng_id, group in df.groupby("engine_id"):
        idx = group.sort_values("cycle").index
        n_cyc = len(idx)
        rul_final = rul_df.loc[eng_id - 1, "RUL"]
        # Cuenta atrás: en el primer ciclo observado quedan
        # (n_cyc - 1 + rul_final) ciclos; en el último quedan rul_final.
        rul_curve = np.arange(n_cyc + rul_final - 1, rul_final - 1, -1)[:n_cyc]
        df.loc[idx, "RUL"] = rul_curve
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    df["early_failure"] = (df["RUL"] <= FAIL_THRESH).astype(int)
    return df


# ─────────────────────────────────────────────
# 3. PENDIENTE OLS VECTORIZADA (convolución)
# ─────────────────────────────────────────────

def ols_slope_vectorized(y: np.ndarray, window: int) -> np.ndarray:
    """
    Pendiente de una regresión OLS de y sobre el tiempo, en ventana
    deslizante de tamaño `window`, calculada por convolución (sin
    bucles Python por posición). Maneja trayectorias más cortas que la
    ventana (frecuente en motores de test truncados tempranamente),
    devolviendo NaN donde no hay suficientes puntos.
    """
    n = len(y)
    if n < 2:
        return np.full(n, np.nan)
    w_eff = min(window, n)
    t = np.arange(w_eff, dtype=np.float64)
    t -= t.mean()
    denom = (t * t).sum()
    if denom == 0:
        return np.full(n, np.nan)
    kernel = t / denom
    conv = np.convolve(y.astype(np.float64), kernel[::-1], mode="valid")
    result = np.full(n, np.nan)
    result[w_eff - 1:] = conv
    return result


# ─────────────────────────────────────────────
# 4. CONSTRUCTOR GENÉRICO DE FEATURES ROLLING
# ─────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, sensor_cols: list,
                          windows: list, stats: list) -> pd.DataFrame:
    """
    Añade, para cada sensor y cada ventana, los estadísticos pedidos en
    `stats` (subconjunto de {"mean","std","min","max","slope"}).

    Base teórica de cada estadístico:
      - mean  : nivel/tendencia central reciente del sensor.
      - std   : variabilidad reciente (ruido vs. señal real).
      - min/max: valores extremos recientes — sensibles a picos de
                 degradación puntuales que la media puede diluir.
      - slope : pendiente OLS — velocidad de cambio, más informativa
                 que el nivel absoluto para detectar aceleración de la
                 degradación (Ramasso & Saxena, 2014).
    """
    out = df.copy()
    for eng_id, group in df.groupby("engine_id"):
        idx = group.sort_values("cycle").index
        for s in sensor_cols:
            vals = df.loc[idx, s].values.astype(np.float64)
            for w in windows:
                ser = pd.Series(vals, index=idx)
                prefix = f"{s}_w{w}"
                if "mean" in stats:
                    out.loc[idx, f"{prefix}_mean"] = ser.rolling(w, min_periods=1).mean().values
                if "std" in stats:
                    out.loc[idx, f"{prefix}_std"] = ser.rolling(w, min_periods=1).std(ddof=1).fillna(0).values
                if "min" in stats:
                    out.loc[idx, f"{prefix}_min"] = ser.rolling(w, min_periods=1).min().values
                if "max" in stats:
                    out.loc[idx, f"{prefix}_max"] = ser.rolling(w, min_periods=1).max().values
                if "slope" in stats:
                    out.loc[idx, f"{prefix}_slope"] = ols_slope_vectorized(vals, w)
    return out


def add_cumulative_delta(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Delta ACUMULADO: Δs(t) = s(t) − s(cycle_inicial_del_motor).
    Captura cuánto se ha desviado el motor de SU PROPIO punto de
    partida — relevante porque, aunque la normalización ahora es
    global (Fase 0), cada motor arranca su vida en un punto propio
    dentro del rango normalizado; esta feature aísla el desplazamiento
    individual de cada trayectoria, no el nivel absoluto del sensor.
    Es la definición usada por el pipeline canónico (112 features),
    reconstruida a partir de fase5_evaluacion.py / fase7_mejoras.py.
    """
    out = df.copy()
    for eng_id, group in df.groupby("engine_id"):
        idx = group.sort_values("cycle").index
        for s in sensor_cols:
            vals = df.loc[idx, s].values.astype(np.float64)
            out.loc[idx, f"{s}_delta"] = vals - vals[0]
    return out


def add_first_order_delta(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Delta de PRIMER ORDEN: Δs(t) = s(t) − s(t−1).
    Proxy discreto de la derivada del sensor — velocidad instantánea de
    cambio, distinta de la pendiente rolling (que suaviza sobre una
    ventana). Es la definición usada por la Fase 2 original (238
    features); se mantiene aquí solo para reproducir esa comparación.
    """
    out = df.copy()
    for eng_id, group in df.groupby("engine_id"):
        idx = group.sort_values("cycle").index
        for s in sensor_cols:
            out.loc[idx, f"{s}_delta"] = df.loc[idx, s].diff().fillna(0).values
    return out


def feature_cols_for(sensor_cols: list, windows: list, stats: list) -> list:
    """
    El vector de features incluye, además de los estadísticos rolling y
    el delta, el valor CRUDO (globalmente normalizado) de cada sensor en
    el ciclo actual: el "nivel" instantáneo es en sí mismo informativo
    (p. ej. una presión ya anómala en el ciclo actual, aunque su media
    móvil todavía no lo refleje), y es la definición que efectivamente
    usan fase5_evaluacion.py / fase7_mejoras.py — el 112 = 14 sensores
    crudos + 14×2×3 rolling + 14 delta del README solo cuadra si se
    incluyen los 14 sensores crudos.
    """
    cols = list(sensor_cols)
    for s in sensor_cols:
        for w in windows:
            for stat in stats:
                cols.append(f"{s}_w{w}_{stat}")
    cols += [f"{s}_delta" for s in sensor_cols]
    return cols


# ─────────────────────────────────────────────
# 5. CONSTRUCCIÓN DE LOS DOS CONJUNTOS DE FEATURES
# ─────────────────────────────────────────────

def build_canonical_112(df: pd.DataFrame, sensor_cols: list) -> tuple:
    """windows=[15,30], mean/std/slope + delta acumulado -> 112 features."""
    out = add_rolling_features(df, sensor_cols, CANONICAL_WINDOWS, CANONICAL_STATS)
    out = add_cumulative_delta(out, sensor_cols)
    cols = feature_cols_for(sensor_cols, CANONICAL_WINDOWS, CANONICAL_STATS)
    return out, cols


def build_extended_238(df: pd.DataFrame, sensor_cols: list) -> tuple:
    """windows=[15,30,50], mean/std/min/max/slope + delta 1er orden -> 238 features."""
    out = add_rolling_features(df, sensor_cols, EXTENDED_WINDOWS, EXTENDED_STATS)
    out = add_first_order_delta(out, sensor_cols)
    cols = feature_cols_for(sensor_cols, EXTENDED_WINDOWS, EXTENDED_STATS)
    return out, cols


# ─────────────────────────────────────────────
# 6. COMPARACIÓN EMPÍRICA (112 vs 238)
# ─────────────────────────────────────────────

def _make_light_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        verbose=-1, random_state=42, n_jobs=-1,
    )


def evaluate_feature_set(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          feat_cols: list, label: str) -> dict:
    """
    Compara conjuntos de features bajo el MISMO protocolo que el resto
    del proyecto usa para comparar modelos (Fase 4/7): GroupKFold(k=5)
    para CV (agrupado por motor, sin fuga de datos entre train/val del
    mismo motor — Li et al., 2018) + reentrenamiento en 100% de train
    y evaluación en el test oficial (último ciclo de cada motor, RUL
    verdadero, capeado a 125). No es el modelo final (eso es Fase 4),
    pero mirar solo CV podría favorecer al conjunto más complejo por
    simple capacidad de ajuste; el test oficial es la comprobación de
    que esa mejora de CV se sostiene fuera de muestra.
    """
    df_clean = train_df.dropna(subset=feat_cols)
    X = df_clean[feat_cols].values.astype(np.float32)
    y = df_clean["RUL"].values.astype(np.float32)
    groups = df_clean["engine_id"].values

    gkf = GroupKFold(n_splits=5)
    t0 = time.time()
    cv_rmse = -cross_val_score(_make_light_lgbm(), X, y, groups=groups, cv=gkf,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    elapsed = time.time() - t0

    # Test oficial: último ciclo de cada motor de test, RUL verdadero.
    test_last = (test_df.dropna(subset=feat_cols + ["RUL"])
                 .sort_values("cycle").groupby("engine_id").tail(1))
    X_test = test_last[feat_cols].values.astype(np.float32)
    y_test = test_last["RUL"].values.astype(np.float32)

    final_model = _make_light_lgbm()
    final_model.fit(X, y)
    y_pred = np.clip(final_model.predict(X_test), 0, RUL_CAP)
    test_rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))

    result = {
        "label": label,
        "n_features": len(feat_cols),
        "n_rows": len(df_clean),
        "cv_rmse_mean": cv_rmse.mean(),
        "cv_rmse_std": cv_rmse.std(),
        "test_rmse": test_rmse,
        "seconds": elapsed,
    }
    return result


# ─────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL — FASE 2
# ─────────────────────────────────────────────

def run_fase2():
    print("\n" + "═" * 70)
    print("  FASE 2 — INGENIERÍA DE CARACTERÍSTICAS (sobre Fase 0)")
    print("═" * 70)

    print("\n[1/6] Cargando salidas de Fase 0...")
    train_norm, test_norm, rul_df, informative = load_fase0_outputs()
    print(f"      Train: {train_norm.shape}  |  Test: {test_norm.shape}"
          f"  |  Sensores informativos: {len(informative)}")

    print("\n[2/6] Calculando target RUL...")
    train_rul = compute_rul_train(train_norm)
    test_rul  = compute_rul_test_full(test_norm, rul_df)
    print(f"      Train — RUL medio: {train_rul['RUL'].mean():.1f}"
          f" | % ciclos en zona de riesgo (RUL<={FAIL_THRESH}): "
          f"{train_rul['early_failure'].mean()*100:.1f}%")

    print("\n[3/6] Construyendo conjunto CANÓNICO "
          f"(windows={CANONICAL_WINDOWS}, stats={CANONICAL_STATS}, delta acumulado)...")
    train_112, cols_112 = build_canonical_112(train_rul, informative)
    test_112,  _        = build_canonical_112(test_rul,  informative)
    print(f"      Features: {len(cols_112)}")

    print("\n[4/6] Construyendo conjunto EXTENDIDO "
          f"(windows={EXTENDED_WINDOWS}, stats={EXTENDED_STATS}, delta 1er orden)...")
    train_238, cols_238 = build_extended_238(train_rul, informative)
    test_238,  _         = build_extended_238(test_rul,  informative)
    print(f"      Features: {len(cols_238)}")

    print("\n[5/6] Comparación empírica (GroupKFold k=5 + test oficial, LightGBM ligero)...")
    res_112 = evaluate_feature_set(train_112, test_112, cols_112, "Canónico (112)")
    print(f"      {res_112['label']:<18} CV RMSE={res_112['cv_rmse_mean']:.3f}"
          f" ± {res_112['cv_rmse_std']:.3f}  | Test RMSE={res_112['test_rmse']:.3f}"
          f"  ({res_112['seconds']:.1f}s)")
    res_238 = evaluate_feature_set(train_238, test_238, cols_238, "Extendido (238)")
    print(f"      {res_238['label']:<18} CV RMSE={res_238['cv_rmse_mean']:.3f}"
          f" ± {res_238['cv_rmse_std']:.3f}  | Test RMSE={res_238['test_rmse']:.3f}"
          f"  ({res_238['seconds']:.1f}s)")

    comparison = pd.DataFrame([res_112, res_238])
    comparison.to_csv(f"{OUTPUT_DIR}/fase2_comparacion_features.csv", index=False)

    cv_winner   = comparison.loc[comparison["cv_rmse_mean"].idxmin(), "label"]
    test_winner = comparison.loc[comparison["test_rmse"].idxmin(), "label"]
    print(f"\n      Mejor CV RMSE  : {cv_winner}")
    print(f"      Mejor Test RMSE: {test_winner}")
    print(f"      -> Fase 4 usa el conjunto CANÓNICO (112) porque es el que "
          f"generó los modelos ya guardados en models/. Si el extendido (238) "
          f"gana en CV pero no de forma consistente en test, es la misma "
          f"conclusión de sesgo-varianza que ya documenta la ablación "
          f"completa de ventanas en fase7 (M6): más features reduce el CV "
          f"pero el beneficio en test es marginal frente al coste de "
          f"complejidad/interpretabilidad.")

    print("\n[6/6] Exportando datasets de features...")
    train_112.to_parquet(f"{OUTPUT_DIR}/train_FD001_features_112.parquet", index=False)
    test_112.to_parquet( f"{OUTPUT_DIR}/test_FD001_features_112.parquet",  index=False)
    train_238.to_parquet(f"{OUTPUT_DIR}/train_FD001_features_238.parquet", index=False)
    test_238.to_parquet( f"{OUTPUT_DIR}/test_FD001_features_238.parquet",  index=False)
    print(f"      train/test _features_112.parquet (canónico, usado en Fase 4)")
    print(f"      train/test _features_238.parquet (comparación/archivo)")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 2")
    print("═" * 70)
    print(comparison.to_string(index=False))
    print("═" * 70)
    print("  ✅ FASE 2 COMPLETADA")
    print("═" * 70 + "\n")

    return train_112, test_112, cols_112, train_238, test_238, cols_238, comparison


if __name__ == "__main__":
    run_fase2()
