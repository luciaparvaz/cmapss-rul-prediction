"""Rutas y constantes globales del proyecto.

Todas las rutas son relativas a la raíz del repo (calculadas con pathlib a
partir de la ubicación de este fichero), nunca absolutas ni de sesión.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "archive"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

for _d in (FIGURES_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
SUBSET = "FD001"

RUL_CAP = 125          # Heimes (2008); piecewise-linear RUL
FAIL_THRESH = 30       # umbral para la variable binaria early_failure
VARIANCE_THRESHOLD = 0.01   # std mínima para considerar un sensor informativo

import os as _os
_SMOKE = _os.environ.get("CMAPSS_SMOKE_TEST") == "1"

N_OUTER_FOLDS = 3 if _SMOKE else 5      # GroupKFold para CV / selección de modelo
N_SEARCH_FOLDS = 3 if _SMOKE else 5     # GroupKFold usado dentro de RandomizedSearchCV
N_SEARCH_ITER = 2 if _SMOKE else 20     # combinaciones de hiperparámetros probadas por modelo

# Candidatas de ventana temporal evaluadas por CV en la Fase de ablación
# (ver ablation.select_windows). No se elige "a ojo": la gana la que tenga
# mejor RMSE medio en GroupKFold CV, con empate resuelto por parsimonia.
WINDOW_CANDIDATES = [
    [15, 30],
    [15, 30, 50],
] if _SMOKE else [
    [15, 30],
    [30, 50],
    [15, 45],
    [15, 30, 45],
    [15, 30, 50],
]

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "background": "#F8FAFC",
    "text": "#1E293B",
}

MODEL_COLORS = {
    "Ridge": PALETTE["neutral"],
    "RandomForest": PALETTE["secondary"],
    "XGBoost": PALETTE["accent"],
    "LightGBM": PALETTE["primary"],
}
