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
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

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

STD_THRESHOLD = 0.01   # umbral de varianza mínima (fase1 original)

# Lista de sensores informativos ya usada consistentemente en fase2,
# fase5 y fase7 (14 sensores tras eliminar 7 con varianza ~nula).
# Se recalcula más abajo y se compara contra esta constante como
# comprobación de consistencia (guardrail ante drift silencioso).
INFORMATIVE_EXPECTED = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
                         's13', 's14', 's15', 's17', 's20', 's21']


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
                                std_threshold: float = STD_THRESHOLD) -> tuple:
    """
    Descarta sensores con std ~0 en train (no aportan señal de
    degradación). El umbral y el criterio se calculan SOLO sobre train,
    nunca sobre test, para no filtrar información del test set en una
    decisión de diseño del pipeline.
    """
    stds = train_df[SENSOR_COLS].std()
    low_var     = stds[stds < std_threshold].index.tolist()
    informative = stds[stds >= std_threshold].index.tolist()

    if informative != INFORMATIVE_EXPECTED:
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
    informative, low_var = select_informative_sensors(train_df)
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


if __name__ == "__main__":
    run_fase0()
