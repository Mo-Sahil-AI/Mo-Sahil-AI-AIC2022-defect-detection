"""
model.py — ResNet18-based binary classifier and EarlyStopping helper.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Load a ResNet18 backbone, freeze early layers, and attach a binary head.

    Strategy:
        - Freeze layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3
        - Fine-tune: layer4, fc (replaced)

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2: ok / defect).
    pretrained : bool
        If True, initialise with ImageNet weights.

    Returns
    -------
    torch.nn.Module
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model   = models.resnet18(weights=weights)
        logger.info("Loaded ResNet18 with ImageNet pre-trained weights.")
    else:
        model = models.resnet18(weights=None)
        logger.info("Loaded ResNet18 with random initialisation.")

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Un-freeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully-connected layer
    in_features = model.fc.in_features   # 512 for ResNet18
    model.fc    = nn.Linear(in_features, num_classes)
    # New layer's parameters have requires_grad=True by default

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model ready — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Usage
    -----
    early_stop = EarlyStopping(patience=3)
    for epoch in ...:
        val_loss = ...
        early_stop(val_loss)
        if early_stop.stop:
            break

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before triggering.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} / {self.patience}"
            )
            if self.counter >= self.patience:
                self.stop = True
                logger.info("Early stopping triggered.")

    def reset(self) -> None:
        """Reset state (useful between cross-validation folds)."""
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False
