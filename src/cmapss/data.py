"""Carga de datos NASA C-MAPSS y cálculo del target RUL.

Referencia: Saxena et al. (2008), "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation", PHM08.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"os{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OS_COLS = ["os1", "os2", "os3"]

# Identificación física de cada sensor (Saxena et al. 2008, Tabla 2).
# s3 y s4 son TEMPERATURAS (T30, T50), no presiones -- error que tenía el
# README anterior y que se corrige aquí como única fuente de verdad.
SENSOR_NAMES = {
    "s1": "T2 - Total temperature at fan inlet (°R)",
    "s2": "T24 - Total temperature at LPC outlet (°R)",
    "s3": "T30 - Total temperature at HPC outlet (°R)",
    "s4": "T50 - Total temperature at LPT outlet (°R)",
    "s5": "P2 - Pressure at fan inlet (psia)",
    "s6": "P15 - Total pressure in bypass-duct (psia)",
    "s7": "P30 - Total pressure at HPC outlet (psia)",
    "s8": "Nf - Physical fan speed (rpm)",
    "s9": "Nc - Physical core speed (rpm)",
    "s10": "epr - Engine pressure ratio (P50/P2)",
    "s11": "Ps30 - Static pressure at HPC outlet (psia)",
    "s12": "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "s13": "NRf - Corrected fan speed (rpm)",
    "s14": "NRc - Corrected core speed (rpm)",
    "s15": "BPR - Bypass ratio",
    "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy",
    "s18": "Nf_dmd - Demanded fan speed (rpm)",
    "s19": "PCNfR_dmd - Demanded corrected fan speed (rpm)",
    "s20": "W31 - HPT coolant bleed (lbm/s)",
    "s21": "W32 - LPT coolant bleed (lbm/s)",
}


def load_cmapss(subset: str = config.SUBSET, data_dir=config.DATA_DIR):
    """Carga train/test/RUL crudos de un subdataset C-MAPSS."""
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_true"])

    for df in (train_df, test_df):
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    return train_df, test_df, rul_df


def select_informative_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                threshold: float = config.VARIANCE_THRESHOLD):
    """Sensores con std >= threshold en TRAIN.

    Se verifica también en test (Bloque 2.9 de la auditoría exigía esto: el
    criterio de fase1 original solo se comprobaba en train). Si un sensor
    difiere de "informativo" entre train y test se reporta pero se conserva
    el criterio de train (es la única partición legítima para decidir qué
    features existen).
    """
    stds_train = train_df[SENSOR_COLS].std()
    stds_test = test_df[SENSOR_COLS].std()

    informative = stds_train[stds_train >= threshold].index.tolist()
    low_var = stds_train[stds_train < threshold].index.tolist()

    mismatch = [s for s in low_var if stds_test[s] >= threshold]
    if mismatch:
        print(f"  [aviso] sensores con std<{threshold} en train pero no en test: {mismatch} "
              "(se descartan igualmente; el criterio se fija en train)")

    return informative, low_var


def compute_rul_train(df: pd.DataFrame, rul_cap: int = config.RUL_CAP) -> pd.DataFrame:
    """RUL = ciclo_max_motor - ciclo_actual, con cap piecewise-linear."""
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL_raw"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)
    df["early_failure"] = (df["RUL_raw"] <= config.FAIL_THRESH).astype(int)
    return df


def compute_rul_test(test_df: pd.DataFrame, rul_df: pd.DataFrame,
                      rul_cap: int = config.RUL_CAP) -> pd.DataFrame:
    """Asigna RUL_true (crudo y capeado) SOLO al último ciclo de cada motor.

    rul_df está indexado 0..N-1 en el mismo orden que los engine_id de test
    (1..N), tal y como documenta el readme oficial de NASA.
    """
    df = test_df.copy()
    last_idx = df.groupby("engine_id")["cycle"].idxmax()

    df["RUL_raw"] = np.nan
    for engine_id, idx in last_idx.items():
        true_rul = rul_df.iloc[engine_id - 1]["RUL_true"]
        df.loc[idx, "RUL_raw"] = true_rul

    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)
    df["censored"] = df["RUL_raw"] >= rul_cap
    return df
