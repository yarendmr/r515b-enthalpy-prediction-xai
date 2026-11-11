"""Reproducible experiment settings, model grids, file paths, and plot colors."""
from pathlib import Path
import os
from itertools import product as iproduct
import numpy as np

SEED = 42
np.random.seed(SEED)

MAVI = "#2E86AB"

KIZIL = "#E84855"

YESIL = "#3BB273"

TURUNCU = "#F4A261"

MOR = "#7A5CFA"

GRI = "#666666"

PROJE_KOKU = Path(__file__).resolve().parents[2]

KLASOR_YOLU = str(PROJE_KOKU / "data" / "raw")

KIZGIN_YOL = os.path.join(KLASOR_YOLU, "R515B_kizgin_buhar.xlsx")

DOYMUS_YOL = os.path.join(KLASOR_YOLU, "R515B_doymus_ozellikleri.xlsx")

CIKTI_KLASORU = str(PROJE_KOKU / "outputs")
os.makedirs(CIKTI_KLASORU, exist_ok=True)

OZELLIKLER = ["T", "P", "v", "s"]

HEDEF = "h"

GB_GRID = {
    "n_estimators": [140, 220],
    "learning_rate": [0.05, 0.07],
    "max_depth": [3, 4],
    "subsample": [0.85, 1.0]
}

RF_GRID = {
    "n_estimators": [200, 320],
    "max_depth": [None, 14],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.8]
}

ET_GRID = {
    "n_estimators": [250, 400],
    "max_depth": [None, 16],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.8]
}

MODEL_GRIDLERI = {"GB": GB_GRID, "RF": RF_GRID, "ET": ET_GRID}

SMOGN_GRID = list(iproduct(
    [3, 5, 7],
    [0.70, 0.80, 0.90],
    [0.001, 0.005, 0.010],
    [False, True]
))
