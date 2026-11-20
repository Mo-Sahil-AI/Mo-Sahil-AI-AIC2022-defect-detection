"""
train.py — Training script for the surface defect detection model.

Usage
-----
python src/train.py \
    --data_dir  data/train \
    --epochs    10 \
    --lr        0.001 \
    --batch_size 32 \
    --output_dir models
"""

import os
import sys
import argparse
import logging
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.config import (
    SEED, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_CLASSES, TRAIN_SPLIT, VAL_SPLIT, DEVICE,
    DATA_DIR, MODEL_SAVE_PATH, RESULTS_DIR, NUM_WORKERS, PIN_MEMORY,
)
from src.utils import set_seed, get_transforms, plot_training_history
from src.model import get_model, EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(data_dir: str, image_size: int):
    """
    Load an ImageFolder dataset, then split into train / val subsets.
    Returns (train_dataset, val_dataset, class_names).
    """
    if not os.path.isdir(data_dir):
        logger.error(f"Training data directory not found at: '{data_dir}'")
        logger.error("Please check where your 'train' folder is located.")
        sys.exit(1)

    # Temporarily load with no transform to perform the split
    full_dataset = ImageFolder(root=data_dir)
    class_names  = full_dataset.classes
    n_total      = len(full_dataset)

    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = n_total - n_train   # absorbs rounding remainder into val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Apply phase-specific transforms by wrapping with a thin dataset
    train_subset.dataset = _TransformDataset(full_dataset, get_transforms("train",  image_size))
    val_subset.dataset   = _TransformDataset(full_dataset, get_transforms("val",    image_size))

    logger.info(
        f"Dataset split — total: {n_total}, train: {n_train}, val: {n_val}"
    )
    logger.info(f"Classes: {class_names}")
    return train_subset, val_subset, class_names


class _TransformDataset(torch.utils.data.Dataset):
    """Wraps an ImageFolder and applies a custom transform at __getitem__."""

    def __init__(self, base_dataset: ImageFolder, transform):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    phase: str,
) -> tuple:
    """Run one forward (and optionally backward) pass over the dataloader.

    Returns
    -------
    (avg_loss, accuracy)  both as Python floats.
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    correct         = 0
    total           = 0

    pbar = tqdm(loader, desc=f"  {phase.capitalize():5s}", leave=False)
    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            _, preds     = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    setup_logging(args.output_dir)
    set_seed(SEED)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Arguments: {vars(args)}")

    # --- Datasets & loaders --------------------------------------------------
    train_ds, val_ds, class_names = build_datasets(args.data_dir, IMAGE_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # --- Model, loss, optimiser, scheduler -----------------------------------
    model     = get_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler     = StepLR(optimizer, step_size=3, gamma=0.1)
    early_stop    = EarlyStopping(patience=3)

    # --- History containers --------------------------------------------------
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_epoch   = 0

    # --- Training loop -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training …")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, "train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, DEVICE, "val"
        )

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"  Train  — loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%"
        )
        logger.info(
            f"  Val    — loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%"
            f"  [{elapsed:.1f}s]"
        )

        # Append history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch":    epoch,
                    "val_acc":  val_acc,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names":          class_names,
                },
                save_path,
            )
            logger.info(f"  [SAVED] Best model saved -> {save_path}  (val_acc={val_acc*100:.2f}%)")

        # Early stopping check
        early_stop(val_loss)
        if early_stop.stop:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # --- Post-training -------------------------------------------------------
    logger.info("=" * 60)
    logger.info(
        f"Training complete. Best val accuracy: {best_val_acc*100:.2f}% "
        f"at epoch {best_epoch}."
    )

    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs, curves_path
    )
    logger.info(f"Training curves saved -> {curves_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the surface defect detection model."
    )
    parser.add_argument("--data_dir",   default=DATA_DIR,       help="Root dir of training data (ImageFolder layout)")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,         help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE,  help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE,     help="Mini-batch size")
    parser.add_argument("--output_dir",             default="models",       help="Directory to save checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
