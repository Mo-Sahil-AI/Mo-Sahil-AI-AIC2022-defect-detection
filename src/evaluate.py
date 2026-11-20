"""
evaluate.py — Evaluate a trained defect detection model on the held-out test set.

Usage
-----
python src/evaluate.py \
    --data_dir  data/test \
    --model     models/best_model.pth \
    --output_dir results
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix,
    classification_report as sk_classification_report,
)
from tqdm import tqdm

from src.config import (
    DEVICE, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    TEST_DATA_DIR, MODEL_SAVE_PATH, RESULTS_DIR,
)
from src.utils  import get_transforms, compute_metrics
from src.model  import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load model weights from a checkpoint produced by train.py."""
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    model       = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_names = checkpoint.get("class_names", [str(i) for i in range(num_classes)])
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
        f"(val_acc={checkpoint.get('val_acc', 0)*100:.2f}%)"
    )
    return model, class_names


def run_inference(model, loader, device) -> tuple:
    """Return (all_preds, all_labels) numpy arrays."""
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images  = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, output_dir: str) -> str:
    """Save a seaborn confusion matrix heatmap and return its path."""
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Defect Detection", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "confusion_matrix.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {out_path}")
    return out_path


def save_classification_report(y_true, y_pred, class_names, output_dir: str) -> str:
    """Save sklearn classification report to a text file."""
    report   = sk_classification_report(y_true, y_pred, target_names=class_names, digits=4)
    out_path = os.path.join(output_dir, "classification_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Classification Report — Surface Defect Detection\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    logger.info(f"Classification report saved -> {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Arguments: {vars(args)}")

    # Dataset
    test_ds = ImageFolder(
        root=args.data_dir,
        transform=get_transforms("test"),
    )
    class_names = test_ds.classes
    logger.info(f"Test samples: {len(test_ds)}  |  Classes: {class_names}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Model
    model, ckpt_classes = load_model(args.model, NUM_CLASSES, DEVICE)
    # Prefer checkpoint class names if available
    if ckpt_classes:
        class_names = ckpt_classes

    # Inference
    y_true, y_pred = run_inference(model, test_loader, DEVICE)

    # Metrics
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())

    print("\n" + "=" * 50)
    print("          EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print("=" * 50 + "\n")

    # Save artefacts
    save_confusion_matrix(y_true, y_pred, class_names, args.output_dir)
    save_classification_report(y_true, y_pred, class_names, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained defect detection model.")
    parser.add_argument("--data_dir",    default=TEST_DATA_DIR,      help="Root dir of test data (ImageFolder layout)")
    parser.add_argument("--model",       default=MODEL_SAVE_PATH,    help="Path to saved checkpoint")
    parser.add_argument("--output_dir",  default=RESULTS_DIR,        help="Where to save evaluation outputs")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
