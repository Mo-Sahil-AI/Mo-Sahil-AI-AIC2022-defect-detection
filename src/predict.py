"""
predict.py — Single-image inference CLI.

Usage
-----
python src/predict.py --image path/to/image.jpg --model models/best_model.pth
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import DEVICE, NUM_CLASSES, CLASS_NAMES, MODEL_SAVE_PATH
from src.utils  import get_transforms
from src.model  import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached if reused in a REPL / notebook)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_classes: int = NUM_CLASSES):
    """Load and return a model from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model      = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Restore class names saved during training (if present)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(image_path: str, model, class_names: list) -> dict:
    """
    Run inference on one image.

    Returns
    -------
    dict with keys:
        label      : str   — predicted class name
        confidence : float — probability of predicted class (0–1)
        probs      : dict  — {class_name: probability} for all classes
    """
    # Load & preprocess
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms("test")
    tensor    = transform(image).unsqueeze(0).to(DEVICE)   # (1, C, H, W)

    # Inference
    with torch.no_grad():
        logits = model(tensor)                             # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)       # (num_classes,)

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    label       = class_names[pred_idx].upper()

    probs_dict = {
        class_names[i].upper(): round(probs[i].item(), 4)
        for i in range(len(class_names))
    }

    return {
        "label":      label,
        "confidence": confidence,
        "probs":      probs_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict defect / ok for a single image."
    )
    parser.add_argument("--image", required=True,                        help="Path to the input image")
    parser.add_argument("--model", default=MODEL_SAVE_PATH,              help="Path to the trained checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    model, class_names = load_model(args.model)
    result             = predict_single(args.image, model, class_names)

    print("\n" + "─" * 40)
    print(f"  Image      : {os.path.basename(args.image)}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print("─" * 40)
    print("  Class probabilities:")
    for cls, prob in result["probs"].items():
        bar = "█" * int(prob * 20)
        print(f"    {cls:10s} {prob*100:6.2f}%  {bar}")
    print("─" * 40 + "\n")
