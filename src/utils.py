"""
utils.py — Utility helpers shared across training, evaluation, and inference.
"""

import os
import random
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torchvision.transforms as T

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN ops (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation statistics
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(phase: str, image_size: int = 224) -> T.Compose:
    """
    Return the appropriate torchvision transform pipeline for the given phase.

    Parameters
    ----------
    phase : str
        One of 'train', 'val', 'test'.
    image_size : int
        Target spatial resolution (default 224 for ResNet).

    Returns
    -------
    torchvision.transforms.Compose
    """
    if phase == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    elif phase in ("val", "test"):
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unknown phase '{phase}'. Choose 'train', 'val', or 'test'.")


# ─────────────────────────────────────────────────────────────────────────────
# Training visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str,
) -> None:
    """
    Plot and save training / validation loss and accuracy curves.

    Parameters
    ----------
    train_losses, val_losses : list of float
        Per-epoch losses.
    train_accs, val_accs : list of float
        Per-epoch accuracies (0–1 range).
    save_path : str
        Full path where the PNG figure will be saved.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    ax.set_title("Loss over Epochs", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in train_accs], "b-o", label="Train Acc", markersize=4)
    ax.plot(epochs, [a * 100 for a in val_accs],   "r-o", label="Val Acc",   markersize=4)
    ax.set_title("Accuracy over Epochs", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Training History — Surface Defect Detection", fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : list of int
        Ground-truth class indices.
    y_pred : list of int
        Predicted class indices.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }
