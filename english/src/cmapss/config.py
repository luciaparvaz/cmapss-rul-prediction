"""Project-wide paths and constants.

All paths are relative to this file's location (computed with pathlib),
never absolute or session-specific.
"""
from pathlib import Path

# This file lives at english/src/cmapss/config.py, three levels below the
# repo root (english/src/cmapss/ -> english/src/ -> english/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[3]
ENGLISH_ROOT = Path(__file__).resolve().parents[2]

# Raw NASA data is shared with the Spanish pipeline (same files, no need to
# duplicate ~5.6MB of identical FD001 data under english/).
DATA_DIR = REPO_ROOT / "archive"

# Everything this pipeline generates lives under english/ so it never
# overwrites the artifacts already committed for the Spanish version.
FIGURES_DIR = ENGLISH_ROOT / "figures"
OUTPUT_DIR = ENGLISH_ROOT / "outputs"
MODEL_DIR = ENGLISH_ROOT / "models"

for _d in (FIGURES_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
SUBSET = "FD001"

RUL_CAP = 125           # Heimes (2008); piecewise-linear RUL
FAIL_THRESH = 30        # threshold for the binary early_failure variable
VARIANCE_THRESHOLD = 0.01   # minimum std to consider a sensor informative

import os as _os
_SMOKE = _os.environ.get("CMAPSS_SMOKE_TEST") == "1"

N_OUTER_FOLDS = 3 if _SMOKE else 5      # GroupKFold for CV / model selection
N_SEARCH_FOLDS = 3 if _SMOKE else 5     # GroupKFold used inside RandomizedSearchCV
N_SEARCH_ITER = 2 if _SMOKE else 20     # hyperparameter combinations tried per model

# Window-size candidates evaluated by CV in the ablation phase (see
# train.py, section 2). Not picked "by eye": the winner is whichever has the
# best mean RMSE in GroupKFold CV, with ties broken by parsimony.
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
