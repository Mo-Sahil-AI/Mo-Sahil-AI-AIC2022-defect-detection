"""
app.py — Gradio web application for Industrial Surface Defect Detection.

Usage
-----
python src/app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from src.config import DEVICE, NUM_CLASSES, MODEL_SAVE_PATH, CLASS_NAMES
from src.utils  import get_transforms
from src.model  import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    if not os.path.isfile(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at '{MODEL_SAVE_PATH}'. "
            "Please run training first:\n"
            "  python src/train.py --data_dir data/train"
        )
    checkpoint  = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model       = get_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    print(
        f"[app] Model loaded from {MODEL_SAVE_PATH}  "
        f"(epoch {checkpoint.get('epoch','?')}, "
        f"val_acc={checkpoint.get('val_acc',0)*100:.2f}%)"
    )
    return model, class_names


_model, _class_names = _load_model()
_transform           = get_transforms("test")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction function (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(pil_image: Image.Image) -> dict:
    """
    Perform inference on a PIL image.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Image uploaded by the user through Gradio.

    Returns
    -------
    dict mapping class label → confidence (for gr.Label).
    """
    if pil_image is None:
        return {"Error: no image provided": 1.0}

    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    tensor = _transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    # Build {label: probability} dict for gr.Label
    result = {
        _class_names[i].upper(): float(probs[i])
        for i in range(len(_class_names))
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────────────────────────────────────

_description = """
## Industrial Surface Defect Detector

Upload an image of a casting product and the model will classify it as:
- **OK** — No surface defect detected
- **DEFECT** — Surface defect detected

*Powered by ResNet18 fine-tuned on the Kaggle Casting Product Defect Dataset.*
"""

_examples_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "test"
)

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload casting product image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Industrial Surface Defect Detector",
    description=_description,
)


if __name__ == "__main__":
    iface.launch(inbrowser=True)
