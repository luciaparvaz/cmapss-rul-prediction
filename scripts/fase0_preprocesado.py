"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 0: Preprocesado canónico (carga + normalización global)
=============================================================================
Este script es la ÚNICA fuente de verdad para dos cosas que, hasta ahora,
vivían duplicadas o rotas en el resto del proyecto:

  1. RUTAS DE DATOS. fase1..fase7 apuntaban a "/home/claude/cmapss/..."
     (ruta del sandbox donde se desarrolló el proyecto originalmente), que
     no existe en este repositorio. Aquí las rutas se calculan de forma
     relativa a la raíz del proyecto, usando archive/ como origen de datos.

  2. NORMALIZACIÓN DE SENSORES. Las versiones antiguas de Fase 1
     (`normalize_per_unit` en fase1_carga_inspeccion.py) ajustan un
     MinMaxScaler POR MOTOR, de forma independiente en train y en test.
     Esto es un look-ahead bias real: para cualquier ciclo t de un motor
     de test, el escalador usa el mínimo/máximo de TODA la trayectoria
     observada de ese motor (incluyendo ciclos posteriores a t), algo que
     no está disponible en un despliegue real en streaming. Además,
     normalizar cada motor de test a su propio rango [0,1] hace que el
     último ciclo observado quede siempre cerca de 1.0 independientemente
     del RUL real, sesgando al modelo a creer que todos los motores de
     test están cerca del fallo.

     Evidencia cuantitativa de este sesgo (ver diagnose_normalization_bias
     más abajo, y fase6_documentacion.py líneas ~488-496): con
     normalización per-motor, s11 en test da μ≈0.72 frente a μ≈0.24 en
     motores sanos de train (RUL≥100). Con normalización global, esa
     diferencia baja a ≈0.03.

     La corrección — un único MinMaxScaler ajustado SOLO sobre el
     conjunto de entrenamiento completo (todos los motores, todos los
     ciclos) y aplicado sin reajuste al test — ya se usaba de forma ad
     hoc, reimplementada por separado, dentro de fase5_evaluacion.py y
     fase7_mejoras.py. Este script la centraliza para que fase1/fase2/
     fase4 puedan (deberían) reescribirse para consumirla, en vez de cada
     script reinventando su propia versión del preprocesado.

Referencias:
  - Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage
    propagation modeling for aircraft engine run-to-failure simulation.
    PHM 2008, Denver, CO.
  - Ramasso, E., & Saxena, A. (2014). Performance benchmarking and
    analysis of prognostic methods for CMAPSS datasets. International
    Journal of Prognostics and Health Management, 5(2).
    DOI: 10.36001/ijphm.2014.v5i2.2236
    (Nota: versiones previas de este proyecto citaban erróneamente esta
    referencia como "NASA/TM-2014-218496", un identificador de informe
    que no corresponde a ningún documento real verificable. La cita
    correcta es la de IJPHM anterior.)

EXTENSIÓN (Fase 1 extendida, sept. 2026): soporte para los 4 subdatasets.
  FD001/FD003 operan a UNA condición operacional (Sea Level) -> el
  GlobalMinMaxScaler de arriba (fit solo en train, sin reajuste en test)
  es correcto tal cual.
  FD002/FD004 combinan SEIS condiciones operacionales distintas (altitud,
  Mach, TRA — ver archive/readme.txt). Normalizar esos dos subdatasets
  con un único scaler global mezclaría en la misma escala mediciones que
  son fisicamente distintas por régimen (p.ej. una temperatura a
  crucero y la misma temperatura a nivel del mar no son comparables sin
  antes des-correlacionar el efecto de la condición operacional). La
  práctica estándar en la literatura (Ramasso & Saxena, 2014; Li, Ding
  & Sun, 2018; Zheng et al., 2017) es agrupar las 3 op_settings en 6
  regímenes vía k-means y normalizar CADA sensor DENTRO de cada régimen
  — nunca por motor, que es precisamente el sesgo que corrigió Fase 0.
  Ver fit_condition_clusters / fit_condition_scalers / normalize_dataset
  más abajo. El pipeline de FD001 (usado por fase2/4/5/7) no se modifica.
=============================================================================
"""

import os
import sys
import warnings

# joblib/loky intenta detectar el nº de núcleos físicos vía un
# subproceso (wmic en Windows) para paralelizar KMeans; en este sandbox
# ese subproceso falla y loky emite un warning con traceback completo
# (inofensivo — no detiene la ejecución, exit 0). Fijar este valor evita
# el intento de detección y el ruido en consola.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
np.random.seed(42)

# La consola de Windows (cp1252) no puede imprimir caracteres como "═" o
# "✅"; forzamos UTF-8 en stdout/stderr para que el script sea portable
# entre Linux (donde se desarrolló originalmente) y Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# RUTAS (relativas a la raíz del proyecto, no al sandbox original)
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(PROJECT_ROOT, "archive")   # aquí viven los .txt de NASA
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
FIG_DIR    = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ─────────────────────────────────────────────
# COLUMNAS Y CONSTANTES
# ─────────────────────────────────────────────
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"os{i}" for i in range(1, 4)]
    + [f"s{i}"  for i in range(1, 22)]
)
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_SETTING_COLS = [f"os{i}" for i in range(1, 4)]

STD_THRESHOLD = 0.01   # umbral de varianza mínima (fase1 original)

# Lista de sensores informativos ya usada consistentemente en fase2,
# fase5 y fase7 (14 sensores tras eliminar 7 con varianza ~nula).
# Se recalcula más abajo y se compara contra esta constante como
# comprobación de consistencia (guardrail ante drift silencioso).
# Válida SOLO para FD001 — cada subdataset tiene su propia lista, ver
# select_informative_sensors(subset=...).
INFORMATIVE_EXPECTED = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
                         's13', 's14', 's15', 's17', 's20', 's21']

# Metadatos oficiales de archive/readme.txt (Saxena et al., 2008) —
# número de condiciones operacionales y modos de fallo por subdataset.
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SUBSET_META = {
    "FD001": {"n_conditions": 1, "n_fault_modes": 1, "fault_desc": "HPC Degradation"},
    "FD002": {"n_conditions": 6, "n_fault_modes": 1, "fault_desc": "HPC Degradation"},
    "FD003": {"n_conditions": 1, "n_fault_modes": 2, "fault_desc": "HPC + Fan Degradation"},
    "FD004": {"n_conditions": 6, "n_fault_modes": 2, "fault_desc": "HPC + Fan Degradation"},
}


# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────

def load_cmapss(subset: str = "FD001", data_dir: str = DATA_DIR) -> tuple:
    """Carga train/test/RUL crudos de NASA C-MAPSS."""
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path  = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{subset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLUMNS)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])

    train_df["engine_id"] = train_df["engine_id"].astype(int)
    train_df["cycle"]     = train_df["cycle"].astype(int)
    test_df["engine_id"]  = test_df["engine_id"].astype(int)
    test_df["cycle"]      = test_df["cycle"].astype(int)

    return train_df, test_df, rul_df


# ─────────────────────────────────────────────
# 2. SELECCIÓN DE SENSORES INFORMATIVOS
# ─────────────────────────────────────────────

def select_informative_sensors(train_df: pd.DataFrame,
                                std_threshold: float = STD_THRESHOLD,
                                subset: str = None) -> tuple:
    """
    Descarta sensores con std ~0 en train (no aportan señal de
    degradación). El umbral y el criterio se calculan SOLO sobre train,
    nunca sobre test, para no filtrar información del test set en una
    decisión de diseño del pipeline.

    El guardrail de consistencia contra INFORMATIVE_EXPECTED (usado por
    fase2/fase5/fase7) solo es válido para FD001: pásalo explícitamente
    en `subset` para activarlo. Para FD002-FD004 el conjunto de sensores
    informativos es distinto (más condiciones operacionales => algunos
    sensores que en FD001 son ruido de instrumentación pasan a tener
    varianza real) y no debe compararse contra la lista de FD001.
    """
    stds = train_df[SENSOR_COLS].std()
    low_var     = stds[stds < std_threshold].index.tolist()
    informative = stds[stds >= std_threshold].index.tolist()

    if subset == "FD001" and informative != INFORMATIVE_EXPECTED:
        warnings.warn(
            "La lista de sensores informativos calculada difiere de la "
            "usada en fase2/fase5/fase7. Verifica std_threshold o si el "
            "dataset ha cambiado.\n"
            f"  Calculada : {informative}\n"
            f"  Esperada  : {INFORMATIVE_EXPECTED}"
        )

    return informative, low_var


# ─────────────────────────────────────────────
# 3. NORMALIZACIÓN GLOBAL (fit SOLO en train)
# ─────────────────────────────────────────────

def fit_global_scaler(train_df: pd.DataFrame, sensor_cols: list) -> MinMaxScaler:
    """
    Ajusta un único MinMaxScaler sobre TODOS los ciclos de TODOS los
    motores de train, por sensor. No se reajusta en test.

    Esto es lo estadísticamente correcto: el escalador es un parámetro
    del modelo (igual que los pesos de una regresión) y debe aprenderse
    exclusivamente de los datos de entrenamiento. Aplicarlo a test sin
    reajustar es la definición misma de "held-out evaluation": el test
    set nunca debe influir en ninguna decisión de preprocesado.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[sensor_cols].values)
    return scaler


def apply_global_scaler(df: pd.DataFrame, sensor_cols: list,
                         scaler: MinMaxScaler) -> pd.DataFrame:
    """Aplica (transform, no fit) el escalador global a un DataFrame."""
    df_scaled = df.copy()
    df_scaled[sensor_cols] = scaler.transform(df[sensor_cols].values)
    return df_scaled


# ─────────────────────────────────────────────
# 4. DIAGNÓSTICO CUANTITATIVO DEL SESGO PER-MOTOR
# ─────────────────────────────────────────────

def diagnose_normalization_bias(train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 sensor_cols: list,
                                 global_scaler: MinMaxScaler,
                                 sensor: str = "s11",
                                 healthy_rul_threshold: int = 100) -> dict:
    """
    Reproduce en código la comparación cuantitativa que hasta ahora solo
    vivía como texto en fase6_documentacion.py: compara, para un sensor
    dado, la media en el ÚLTIMO ciclo de cada motor de test bajo:

      (a) normalización per-motor (MinMaxScaler ajustado en cada motor
          de test de forma independiente — el bug), vs.
      (b) normalización global (este script — el fix), vs.
      (c) la misma media en motores SANOS de train (RUL_raw >= threshold),
          como referencia de "estado no degradado".

    Si (a) está muy por encima de (c) mientras (b) está cerca de (c),
    queda demostrado numéricamente que la normalización per-motor infla
    artificialmente la señal de "cerca del fallo" en todo el test set,
    independientemente del RUL real.
    """
    # (a) Normalización per-motor en test (el bug, replicado aquí solo
    #     para diagnóstico — no se usa para modelar)
    per_unit_last = []
    for eng_id, group in test_df.groupby("engine_id"):
        vals = group[[sensor]].values.astype(np.float64)
        smin, smax = vals.min(), vals.max()
        denom = (smax - smin) if smax > smin else 1.0
        scaled_last = (vals[-1, 0] - smin) / denom
        per_unit_last.append(scaled_last)
    mean_per_unit_test = float(np.mean(per_unit_last))

    # (b) Normalización global en test (último ciclo de cada motor)
    test_global = apply_global_scaler(test_df, sensor_cols, global_scaler)
    last_idx = test_global.groupby("engine_id")["cycle"].idxmax()
    mean_global_test = float(test_global.loc[last_idx, sensor].mean())

    # (c) Referencia: motores sanos de train bajo normalización global
    train_global = apply_global_scaler(train_df, sensor_cols, global_scaler)
    train_global["rul_raw"] = (train_df.groupby("engine_id")["cycle"]
                                .transform("max") - train_df["cycle"])
    healthy_mask = train_global["rul_raw"] >= healthy_rul_threshold
    mean_healthy_train = float(train_global.loc[healthy_mask, sensor].mean())

    result = {
        "sensor": sensor,
        "mean_per_unit_test_bug":     mean_per_unit_test,
        "mean_global_test_fixed":     mean_global_test,
        "mean_healthy_train_reference": mean_healthy_train,
        "gap_bug":   abs(mean_per_unit_test - mean_healthy_train),
        "gap_fixed": abs(mean_global_test   - mean_healthy_train),
    }
    return result


# ─────────────────────────────────────────────
# 4b. NORMALIZACIÓN POR CONDICIÓN OPERACIONAL (FD002 / FD004)
# ─────────────────────────────────────────────

def fit_condition_clusters(train_df: pd.DataFrame, n_conditions: int = 6,
                            random_state: int = 42) -> KMeans:
    """
    Agrupa los 3 op_settings de train en `n_conditions` regímenes vía
    k-means (n_conditions=6 es el valor documentado en archive/readme.txt
    para FD002/FD004 y el usado en la literatura — Ramasso & Saxena,
    2014; Li, Ding & Sun, 2018). Se ajusta SOLO en train, igual que el
    scaler global de FD001/FD003: la asignación de régimen de un ciclo
    de test se hace con este modelo ya entrenado (predict, no fit).
    """
    kmeans = KMeans(n_clusters=n_conditions, random_state=random_state, n_init=10)
    kmeans.fit(train_df[OP_SETTING_COLS].values)
    return kmeans


def assign_conditions(df: pd.DataFrame, kmeans: KMeans) -> pd.Series:
    """Asigna a cada fila su régimen operacional (predict, nunca fit)."""
    return pd.Series(kmeans.predict(df[OP_SETTING_COLS].values),
                      index=df.index, name="op_condition")


def fit_condition_scalers(train_df: pd.DataFrame, sensor_cols: list,
                           kmeans: KMeans) -> dict:
    """
    Ajusta un MinMaxScaler independiente POR RÉGIMEN operacional, cada
    uno SOLO sobre las filas de train que caen en ese régimen. Esto es
    distinto de normalizar por motor (el bug de la Fase 1 original): un
    régimen agrupa ciclos de MUCHOS motores distintos que comparten
    condición física, no la trayectoria completa de un único motor, así
    que no hay look-ahead (ningún ciclo usa información de "su propio
    futuro") ni fuga de la señal de degradación.
    """
    conditions = assign_conditions(train_df, kmeans)
    scalers = {}
    for cond_id in sorted(conditions.unique()):
        mask = conditions == cond_id
        scaler = MinMaxScaler()
        scaler.fit(train_df.loc[mask, sensor_cols].values)
        scalers[cond_id] = scaler
    return scalers


def apply_condition_scalers(df: pd.DataFrame, sensor_cols: list,
                             kmeans: KMeans, scalers: dict) -> pd.DataFrame:
    """Asigna régimen (predict) y aplica (transform) el scaler de ese régimen."""
    df_scaled = df.copy()
    conditions = assign_conditions(df, kmeans)
    df_scaled["op_condition"] = conditions
    for cond_id, scaler in scalers.items():
        mask = conditions == cond_id
        if mask.any():
            df_scaled.loc[mask, sensor_cols] = scaler.transform(df.loc[mask, sensor_cols].values)
    return df_scaled


def normalize_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       sensor_cols: list, subset: str,
                       n_conditions: int = 6, random_state: int = 42) -> tuple:
    """
    Dispatcher de normalización según el nº de condiciones operacionales
    del subdataset (metadatos oficiales en SUBSET_META):

      - FD001 / FD003 (1 condición) -> GlobalMinMaxScaler (fit_global_scaler).
      - FD002 / FD004 (6 condiciones) -> un MinMaxScaler por régimen,
        asignado vía k-means fit en train (fit_condition_scalers).

    Ambas estrategias comparten el mismo principio que corrigió Fase 0:
    todo parámetro de normalización se aprende EXCLUSIVAMENTE en train y
    se aplica a test sin reajuste (held-out evaluation). Nunca se
    normaliza por motor.

    Retorna
    -------
    train_norm, test_norm, info : dict con la estrategia usada y los
    objetos ajustados (scaler único, o kmeans + dict de scalers).
    """
    n_cond_meta = SUBSET_META.get(subset, {}).get("n_conditions", 1)

    if n_cond_meta <= 1:
        scaler = fit_global_scaler(train_df, sensor_cols)
        train_norm = apply_global_scaler(train_df, sensor_cols, scaler)
        test_norm  = apply_global_scaler(test_df,  sensor_cols, scaler)
        info = {"strategy": "global", "scaler": scaler}
    else:
        kmeans = fit_condition_clusters(train_df, n_conditions=n_conditions,
                                         random_state=random_state)
        scalers = fit_condition_scalers(train_df, sensor_cols, kmeans)
        train_norm = apply_condition_scalers(train_df, sensor_cols, kmeans, scalers)
        test_norm  = apply_condition_scalers(test_df,  sensor_cols, kmeans, scalers)
        info = {"strategy": "per_condition", "kmeans": kmeans, "scalers": scalers}

    return train_norm, test_norm, info


# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL — FASE 0
# ─────────────────────────────────────────────

def run_fase0():
    print("\n" + "═" * 70)
    print("  FASE 0 — PREPROCESADO CANÓNICO (carga + normalización global)")
    print("  NASA C-MAPSS | Subdataset FD001")
    print("═" * 70)

    print("\n[1/5] Cargando datos crudos desde:", DATA_DIR)
    train_df, test_df, rul_df = load_cmapss("FD001")
    print(f"      Train : {train_df.shape}  |  Test : {test_df.shape}"
          f"  |  RUL : {rul_df.shape}")

    print("\n[2/5] Seleccionando sensores informativos (std >= "
          f"{STD_THRESHOLD} en train)...")
    informative, low_var = select_informative_sensors(train_df, subset="FD001")
    print(f"      Informativos ({len(informative)}): {informative}")
    print(f"      Descartados  ({len(low_var)}): {low_var}")

    print("\n[3/5] Ajustando GlobalMinMaxScaler SOLO sobre train...")
    global_scaler = fit_global_scaler(train_df, informative)
    joblib.dump(global_scaler, f"{MODEL_DIR}/global_sensor_scaler.pkl")
    print(f"      Guardado: {MODEL_DIR}/global_sensor_scaler.pkl")

    print("\n[4/5] Diagnóstico cuantitativo del sesgo per-motor vs global...")
    diag = diagnose_normalization_bias(train_df, test_df, informative,
                                        global_scaler, sensor="s11")
    print(f"      Sensor analizado           : {diag['sensor']}")
    print(f"      Media test (per-motor, bug): {diag['mean_per_unit_test_bug']:.3f}")
    print(f"      Media test (global, fix)   : {diag['mean_global_test_fixed']:.3f}")
    print(f"      Media train sano (ref.)    : {diag['mean_healthy_train_reference']:.3f}")
    print(f"      |gap| con normalización bug: {diag['gap_bug']:.3f}")
    print(f"      |gap| con normalización fix: {diag['gap_fixed']:.3f}")
    if diag["gap_fixed"] < diag["gap_bug"]:
        print("      -> Confirmado: la normalización global reduce el sesgo "
              "frente a la per-motor.")
    else:
        print("      -> ADVERTENCIA: el diagnóstico no confirma la reducción "
              "de sesgo esperada. Revisar antes de continuar.")

    print("\n[5/5] Aplicando escalador global y exportando...")
    train_norm = apply_global_scaler(train_df, informative, global_scaler)
    test_norm  = apply_global_scaler(test_df,  informative, global_scaler)
    train_norm.to_parquet(f"{OUTPUT_DIR}/train_FD001_normalized.parquet", index=False)
    test_norm.to_parquet( f"{OUTPUT_DIR}/test_FD001_normalized.parquet",  index=False)
    rul_df.to_parquet(    f"{OUTPUT_DIR}/RUL_FD001.parquet",              index=False)
    print(f"      train_FD001_normalized.parquet : {train_norm.shape}")
    print(f"      test_FD001_normalized.parquet  : {test_norm.shape}")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 0")
    print("═" * 70)
    print(f"  Sensores informativos   : {len(informative)}")
    print(f"  Normalización           : MinMaxScaler GLOBAL (fit solo en train)")
    print(f"  Scaler guardado         : models/global_sensor_scaler.pkl")
    print(f"  Sesgo per-motor (gap)   : {diag['gap_bug']:.3f}  ->  "
          f"corregido a {diag['gap_fixed']:.3f}")
    print("═" * 70)
    print("  ✅ FASE 0 COMPLETADA")
    print("═" * 70 + "\n")

    return train_df, test_df, rul_df, train_norm, test_norm, informative, global_scaler


# ─────────────────────────────────────────────
# 6. PIPELINE MULTI-SUBDATASET — FD001-FD004
# ─────────────────────────────────────────────

def run_fase0_all_subsets(subsets: list = SUBSETS) -> dict:
    """
    Repite el preprocesado canónico para los 4 subdatasets, eligiendo
    la estrategia de normalización correcta para cada uno según su
    número de condiciones operacionales (normalize_dataset):

      FD001, FD003 -> GlobalMinMaxScaler   (1 condición, idéntico a run_fase0)
      FD002, FD004 -> MinMaxScaler por régimen k-means (6 condiciones)

    No repite el pipeline de FD001 vía run_fase0() (para no reajustar
    dos veces el mismo scaler); en su lugar usa normalize_dataset con
    la misma función fit_global_scaler subyacente, así que el resultado
    para FD001 es idéntico al de run_fase0(). Los artefactos de FD001
    ya guardados por run_fase0() (global_sensor_scaler.pkl,
    train/test_FD001_normalized.parquet) no se sobrescriben con
    contenido distinto.

    Retorna
    -------
    dict {subset: {"train_raw", "test_raw", "rul", "train_norm",
                   "test_norm", "informative", "low_var", "norm_info"}}
    """
    print("\n" + "═" * 70)
    print("  FASE 0 (extendida) — LOS 4 SUBDATASETS C-MAPSS")
    print("═" * 70)

    results = {}
    for subset in subsets:
        meta = SUBSET_META[subset]
        print(f"\n── {subset} "
              f"({meta['n_conditions']} condición(es), "
              f"{meta['n_fault_modes']} modo(s) de fallo: {meta['fault_desc']}) "
              + "─" * max(1, 40 - len(subset)))

        train_df, test_df, rul_df = load_cmapss(subset)
        print(f"   Train : {train_df.shape}  |  Test : {test_df.shape}  |  RUL : {rul_df.shape}"
              f"  |  Motores train/test: {train_df['engine_id'].nunique()}/"
              f"{test_df['engine_id'].nunique()}")

        informative, low_var = select_informative_sensors(train_df, subset=subset)
        print(f"   Sensores informativos ({len(informative)}): {informative}")
        print(f"   Sensores descartados  ({len(low_var)}): {low_var}")

        strategy = "GlobalMinMaxScaler" if meta["n_conditions"] == 1 else "MinMaxScaler por régimen (k-means, k=6)"
        print(f"   Normalización: {strategy}, fit SOLO en train")
        train_norm, test_norm, norm_info = normalize_dataset(
            train_df, test_df, informative, subset=subset
        )

        if norm_info["strategy"] == "per_condition":
            cond_counts = assign_conditions(train_df, norm_info["kmeans"]).value_counts().sort_index()
            print(f"   Distribución de ciclos de train por régimen: {cond_counts.to_dict()}")
            joblib.dump(norm_info["kmeans"], f"{MODEL_DIR}/condition_kmeans_{subset}.pkl")
            joblib.dump(norm_info["scalers"], f"{MODEL_DIR}/condition_scalers_{subset}.pkl")
        else:
            joblib.dump(norm_info["scaler"], f"{MODEL_DIR}/global_sensor_scaler_{subset}.pkl")

        train_norm.to_parquet(f"{OUTPUT_DIR}/train_{subset}_normalized.parquet", index=False)
        test_norm.to_parquet( f"{OUTPUT_DIR}/test_{subset}_normalized.parquet",  index=False)
        rul_df.to_parquet(    f"{OUTPUT_DIR}/RUL_{subset}.parquet",              index=False)
        print(f"   Exportado: train/test_{subset}_normalized.parquet, RUL_{subset}.parquet")

        results[subset] = {
            "train_raw": train_df, "test_raw": test_df, "rul": rul_df,
            "train_norm": train_norm, "test_norm": test_norm,
            "informative": informative, "low_var": low_var,
            "norm_info": norm_info,
        }

    print("\n" + "═" * 70)
    print("  RESUMEN — 4 SUBDATASETS")
    print("═" * 70)
    summary_rows = []
    for subset in subsets:
        r = results[subset]
        summary_rows.append({
            "Subdataset": subset,
            "Motores train": r["train_raw"]["engine_id"].nunique(),
            "Motores test": r["test_raw"]["engine_id"].nunique(),
            "Cond. operacionales": SUBSET_META[subset]["n_conditions"],
            "Modos de fallo": SUBSET_META[subset]["n_fault_modes"],
            "Sensores informativos": len(r["informative"]),
            "Normalización": r["norm_info"]["strategy"],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Subdataset")
    print(summary_df.to_string())
    summary_df.to_csv(f"{OUTPUT_DIR}/fase0_subsets_summary.csv")
    print("═" * 70)
    print("  ✅ FASE 0 (4 subdatasets) COMPLETADA")
    print("═" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_fase0()
    run_fase0_all_subsets()
