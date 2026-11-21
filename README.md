# 🔍 Industrial Surface Defect Detection System

[![Course](https://img.shields.io/badge/Course-AIC2022%20Introduction%20to%20AI-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.14-green.svg)](#)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-red.svg)](#)
[![UI](https://img.shields.io/badge/Gradio-Web%20UI-orange.svg)](#)
[![Test%20Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.58%25-brightgreen.svg)](#)

An intelligent, deep-learning-based visual quality control system for automated industrial surface defect inspection on cast pump impellers. Built for **Course Assignment AIC2022: Develop an Intelligent Software Application**.

---

## 📌 Executive Summary

Manual surface inspection in modern high-throughput manufacturing is constrained by human fatigue, subjective grading, and high operational costs. This software system implements an automated computer vision inspection pipeline using **Transfer Learning with ResNet-18**. 

Key Highlights:
- **Test Accuracy:** **99.58%** on 715 held-out test specimens.
- **Defect Precision:** **100.00%** (Zero false alarms on non-defective parts).
- **Defect Recall:** **99.34%** (Only 3 missed defects out of 453).
- **Interactive UI:** Real-time drag-and-drop web application powered by **Gradio**.
- **CPU & GPU Ready:** Trains efficiently in under 20 minutes on standard laptop CPUs.

---

## 📁 Repository Structure

```
defect_detection/
├── demo_video/                       # Application demonstration video
│   └── app_demonstration.mp4         # 1:27 min live walkthrough of the Gradio Web App
├── latex_report/                     # Complete Overleaf-ready LaTeX project
│   ├── AIC2022_LaTeX_Report.pdf      # Typeset IEEE technical report (18 pages)
│   ├── main.tex                      # Full LaTeX source code
│   └── references.bib                # 15 IEEE BibTeX citations
├── models/                           # Model checkpoints & logs
│   ├── best_model.pth                # Fine-tuned ResNet-18 weights (val_acc=99.65%)
│   └── training.log                  # Comprehensive training and validation log
├── notebooks/
│   └── demo.ipynb                    # Step-by-step interactive Jupyter notebook
├── report/                           # Markdown documentation & report files
│   ├── AIC2022_LaTeX_Report.pdf      # Compiled publication-grade PDF report
│   ├── AIC2022_Report.md             # Complete technical report source text
│   └── cover_page.md                 # Assignment cover page template
├── results/                          # Evaluation outputs & visual figures
│   ├── classification_report.txt     # Per-class precision, recall, and F1 metrics
│   ├── confusion_matrix.png          # Visual confusion matrix heatmap
│   └── training_curves.png           # Loss and accuracy progression curves
├── src/                              # Core Python application modules
│   ├── __init__.py                   # Package initialiser
│   ├── app.py                        # Gradio interactive web interface
│   ├── config.py                     # Centralized hyperparameters & auto-path resolution
│   ├── evaluate.py                   # Quantitative model evaluation script
│   ├── model.py                      # ResNet-18 architecture & EarlyStopping logic
│   ├── predict.py                    # Single-image command-line inference
│   ├── train.py                      # Training loop with StepLR & validation
│   └── utils.py                      # Data transforms, metrics & plotting helpers
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## 🗃️ Dataset Specification

This application uses the **Real Life Industrial Dataset of Casting Product** (available publicly on [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)):
- **Total Images:** 7,348 industrial casting images (greyscale / RGB).
- **Classes:** 
  - `def_front`: Defective casting surfaces (porosity, shrinkage, inclusions).
  - `ok_front`: High-integrity, defect-free components.
- **Splits:** 
  - Training + Validation: 6,633 samples (4,643 train / 1,990 val).
  - Held-out Test Set: 715 samples (453 defect / 262 ok).
- **Smart Path Auto-Detection:** The project's `src/config.py` automatically detects the dataset whether placed in `Downloads/archive/casting_data` or the project root `data/`.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+ (Tested up to Python 3.14 on Windows 11)
- Git (optional)

### 2. Environment Setup

```powershell
# Navigate into the project folder
cd C:\Users\kas20\defect_detection

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### Step 1: Model Training
Train the fine-tuned ResNet-18 classifier (with automated early stopping):

```powershell
python src/train.py
```
* **Defaults:** 10 epochs (early stops when converged), batch size 32, Adam optimizer (lr=0.001), StepLR scheduler.
* **Outputs:** Saves the top-performing checkpoint to `models/best_model.pth` and visual training curves to `results/training_curves.png`.

### Step 2: Model Evaluation
Evaluate the trained model against the unseen test dataset:

```powershell
python src/evaluate.py
```
* **Outputs:** Prints full metrics table in the console and saves `results/confusion_matrix.png` and `results/classification_report.txt`.

### Step 3: Launch Interactive Web Application
Launch the operator web interface:

```powershell
python src/app.py
```
* Automatically opens your default web browser at **`http://127.0.0.1:7860`**.
* Allows drag-and-drop image upload with instantaneous confidence scoring.

### Step 4: Single-Image CLI Prediction
Test any individual image directly from PowerShell:

```powershell
python src/predict.py --image "path\to\casting_image.jpeg"
```

---

## 📊 Experimental Results

### 1. Test Performance Metrics (715 Test Specimens)

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Defective (`def_front`)** | **1.0000** | **0.9934** | **0.9967** | 453 |
| **Normal (`ok_front`)** | **0.9887** | **1.0000** | **0.9943** | 262 |
| **Weighted Average** | **0.9959** | **0.9958** | **0.9958** | **715** |
| **Overall Accuracy** | — | — | **99.58%** | **715** |

### 2. Confusion Matrix

```
                      PREDICTED
                 OK_FRONT    DEF_FRONT
ACTUAL  OK_FRONT    262          0        (100.0% Specificity)
       DEF_FRONT      3        450        ( 99.3% Sensitivity)
```

---

## 🧠 Architectural Overview

The AI model leverages transfer learning from a ResNet-18 convolutional backbone pretrained on the ImageNet-1K benchmark:

```
[Input Image: 224x224x3]
         │
         ▼
[Data Augmentations: RandomHorizontalFlip, RandomVerticalFlip, Rotation(15°), ColorJitter]
         │
         ▼
[ResNet-18 Backbone (conv1 through layer3: Frozen Feature Extractors)]
         │
         ▼
[ResNet-18 layer4: Trainable High-Level Representation Filters]
         │
         ▼
[Adaptive Average Pooling (512x1x1)] ➔ [Flatten (512-dim)]
         │
         ▼
[Custom Classification Head: Linear(512 -> 2)]
         │
         ▼
[Softmax Activation] ➔ [OK vs DEFECT Prediction + Confidence Score]
```

---

## 📝 Academic Submission Guidelines (AIC2022)

For submission via Google Classroom:

1. **Technical Report:**
   - Full 10–15 page report in standard IEEE structure is provided in [`report/AIC2022_Report.md`](report/AIC2022_Report.md).
   - Edit [`report/cover_page.md`](report/cover_page.md) with your student information.
   - Insert figures from `results/` (`training_curves.png`, `confusion_matrix.png`) and export as PDF.
2. **Source Code Submission:**
   - Compress the `defect_detection` directory into a `.zip` archive (ensure `models/best_model.pth` is included).

---

## 📜 License

Created for educational purposes under the **AIC2022: Introduction to Artificial Intelligence** course. All code and documentation are available for academic review.
