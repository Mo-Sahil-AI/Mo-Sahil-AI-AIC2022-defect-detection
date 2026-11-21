"""
config.py — Centralised configuration for all hyperparameters and paths.
All other modules import from here so changes propagate everywhere.
"""

import os
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Image settings ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224          # ResNet18 expects 224x224
BATCH_SIZE = 32

# ── Training settings ────────────────────────────────────────────────────────
EPOCHS        = 10
LEARNING_RATE = 0.001
NUM_CLASSES   = 2

# Dataset split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ── Class mapping ─────────────────────────────────────────────────────────────
# The Kaggle casting dataset uses folder names "def_front" and "ok_front".
# ImageFolder sorts alphabetically → "def_front"=0, "ok_front"=1
CLASS_NAMES = ["def_front", "ok_front"]   # index 0 → defective, index 1 → ok

# ── Paths ────────────────────────────────────────────────────────────────────
# Auto-detect dataset root (handles nested archive/casting_data/casting_data structures)
def _find_dataset_root() -> str:
    candidates = [
        # Linux / Ubuntu / Mac paths
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        os.path.expanduser("~/Downloads/archive/casting_data/casting_data"),
        os.path.expanduser("~/Downloads/archive/casting_data"),
        os.path.expanduser("~/Downloads/archive"),
        os.path.expanduser("~/archive/casting_data/casting_data"),
        os.path.expanduser("~/archive/casting_data"),
        # Windows paths
        r"C:\Users\kas20\Downloads\archive\casting_data\casting_data",
        r"C:\Users\kas20\Downloads\archive\casting_data",
        r"C:\Users\kas20\Downloads\archive",
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train")):
            return candidate

    search_base = r"C:\Users\kas20\Downloads\archive"
    if os.path.exists(search_base):
        for root, dirs, _ in os.walk(search_base):
            if "train" in dirs:
                return root

    return r"C:\Users\kas20\Downloads\archive\casting_data"

DATASET_ROOT    = _find_dataset_root()
DATA_DIR        = os.path.join(DATASET_ROOT, "train")
TEST_DATA_DIR   = os.path.join(DATASET_ROOT, "test")

# Resolve project root (one level above src/) for saving outputs
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

MODEL_SAVE_PATH = os.path.join(_ROOT_DIR, "models", "best_model.pth")
RESULTS_DIR     = os.path.join(_ROOT_DIR, "results")

# Ensure output directories exist
os.makedirs(os.path.join(_ROOT_DIR, "models"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0    # Set to 0 on Windows to avoid multiprocessing issues
PIN_MEMORY  = torch.cuda.is_available()
