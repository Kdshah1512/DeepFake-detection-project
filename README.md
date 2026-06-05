# DeepFake-detection-project
# Explainable Deepfake Detection Using CLIP and Eye-Blink Dynamics

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.5.0-purple.svg)](https://lightning.ai/)
[![License: Academic](https://img.shields.io/badge/License-Academic_Use_Only-green.svg)]()

> **MSc Data Science and Artificial Intelligence — Master's Thesis**  
> GISMA University of Applied Sciences · Module M599 · 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Research Contributions](#2-research-contributions)
3. [Results at a Glance](#3-results-at-a-glance)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Dataset Preparation](#6-dataset-preparation)
7. [CLIP Pipeline — Training and Evaluation](#7-clip-pipeline--training-and-evaluation)
8. [EAR Pipeline — Original Contribution](#8-ear-pipeline--original-contribution)
9. [Reproducing Published Results](#9-reproducing-published-results)
10. [Full Results Table](#10-full-results-table)
11. [Configuration Reference](#11-configuration-reference)
12. [Dataset Licences](#12-dataset-licences)
13. [Citation](#13-citation)
14. [Acknowledgements](#14-acknowledgements)

---

## 1. Overview

This repository contains the complete implementation of a parameter-efficient deepfake detection system built on OpenAI's **CLIP Vision Transformer** encoder, extended with an original **Eye Aspect Ratio (EAR) physiological blink feature extractor**.

The system trains exclusively on **FaceForensics++** and cross-evaluates on **Celeb-DF-v2 (CDFv2)** — a completely independent dataset with different subjects and forgery techniques — with no target-domain fine-tuning at any stage.

**Two independent pipelines are implemented and compared:**

| Pipeline | Method | Video AUC-ROC on CDFv2 |
|---|---|---|
| CLIP ViT-B/32 | LN-Tuning on CLIP | 0.9991 |
| CLIP ViT-B/16 | LN-Tuning on CLIP | 0.9962 |
| **CLIP ViT-L/14 ★** | **LN-Tuning on CLIP** | **0.9984** |
| EAR baseline | 9-dim blink features + Logistic Regression | 0.7866 |

★ Primary model — best frame-level AUC-ROC (0.9753) and video-level balanced accuracy (0.8938). All three CLIP configurations achieve **100% fake recall** at the video level.

The **21.2 percentage point gap** between the best CLIP model (0.9984) and the EAR baseline (0.7866) on CDFv2 is the central empirical finding: CLIP visual embeddings substantially outperform physiological blink dynamics as a standalone cross-dataset detection signal.

---

## 2. Research Contributions

1. **Parameter-efficient CLIP adaptation** — Layer Normalisation Tuning (LN-Tuning) updates fewer than 0.01% of backbone parameters while achieving state-of-the-art cross-dataset generalisation on CDFv2.

2. **Three-way backbone comparison** — ViT-B/16, ViT-B/32, and ViT-L/14 evaluated under identical conditions on the FF++ → CDFv2 protocol, providing empirical guidance on the accuracy–efficiency trade-off.

3. **Original EAR feature extractor** — A fully implemented standalone pipeline that extracts a nine-dimensional statistical blink descriptor per video using 68-point facial landmark detection. This module is **not present in the baseline codebase** and is the original contribution of this dissertation.

4. **First direct CLIP vs EAR comparison** — The first published empirical comparison of deep CLIP visual embeddings against physiological EAR blink dynamics as competing standalone detection signals on a strict cross-dataset evaluation.

5. **Explainability framework** — SHAP-based post-hoc explanation designed for the lightweight classifier stage, with interpretable EAR features providing human-readable blink dynamics alongside opaque CLIP embeddings.

---

## 3. Results at a Glance

### CLIP Pipeline (CDFv2 cross-dataset test)

| Metric | ViT-B/32 | ViT-B/16 | ViT-L/14 |
|---|---|---|---|
| Frame AUC-ROC | 0.9470 | 0.9629 | **0.9753** |
| Frame Balanced Acc. | 0.7711 | 0.8280 | **0.8469** |
| Frame F1 | 0.7029 | 0.7794 | **0.8019** |
| Video AUC-ROC | **0.9991** | 0.9962 | 0.9984 |
| Video Balanced Acc. | 0.7750 | 0.8688 | **0.8938** |
| Video F1 | 0.6997 | 0.8205 | **0.8529** |
| Fake Recall (Video) | 100% | 100% | 100% |
| Real FP Rate (Video) | 45.00% | 26.25% | **21.25%** |

### EAR Pipeline (standalone physiological baseline)

| Metric | FF++ (in-distribution) | CDFv2 (cross-dataset) |
|---|---|---|
| AUC-ROC | 0.4871 | 0.7866 |
| Balanced Accuracy | 0.5076 | 0.7375 |
| F1 Score | 0.1157 | 0.6517 |
| Fake Recall | 6.23% | 72.50% |

---

## 4. Repository Structure

```
DeepFake-detection-project/
│
├── train.py                    # CLIP pipeline — training and testing entry point
├── ear_baseline.py             # EAR pipeline — extract, train, evaluate (original)
├── perpare_dataset.py          # Face extraction (MTCNN) — shared by both pipelines
├── download.py                 # FaceForensics++ official downloader
├── data_conversion.sh          # CDFv2 manifest generation
├── req.txt                     # Python dependencies
│
├── src/
│   ├── config.py               # Pydantic configuration system
│   ├── loss.py                 # Cross-entropy + alignment + uniformity losses
│   ├── metrics.py              # AUC-ROC, mAP, EER computation
│   ├── plots.py                # ROC, PRC, F1, confusion matrix plots
│   │
│   ├── encoders/
│   │   └── clip_encoder.py     # CLIPEncoder and CLIPEncoderPatches
│   │
│   ├── heads/
│   │   └── head.py             # LinearProbe and LinearNorm heads
│   │
│   ├── model/
│   │   └── dfdet.py            # DeepfakeDetectionModel (PyTorch Lightning module)
│   │
│   ├── dataset/
│   │   ├── base.py             # BaseDataset and BaseDataModule
│   │   └── deepfake.py         # DeepfakeDataset and DeepfakeDataModule
│   │
│   ├── losses/
│   │   └── unifalign.py        # Alignment and uniformity loss functions
│   │
│   └── ear/                    # ★ EAR module — original contribution
│       ├── __init__.py
│       ├── extractor.py        # 68-point landmark detection, EAR computation
│       ├── classifier.py       # StandardScaler + LogisticRegression pipeline
│       └── evaluate.py         # Metrics, ROC/CM plots, JSON output
│
└── config/
    └── datasets/
        ├── FF/test/            # FaceForensics++ image path manifests
        │   ├── DF.txt
        │   ├── F2F.txt
        │   ├── FS.txt
        │   ├── NT.txt
        │   └── real.txt
        └── CDFv2/test/         # Celeb-DF-v2 image path manifests
            ├── Celeb-synthesis.txt
            ├── Celeb-real.txt
            └── YouTube-real.txt
```

---

## 5. Installation

### Step 1 — Clone

```bash
git clone https://github.com/Kdshah1512/DeepFake-detection-project.git
cd DeepFake-detection-project
```

### Step 2 — Create Environment

```bash
conda create -n dfdet python=3.12
conda activate dfdet
```

### Step 3 — Install Dependencies

```bash
pip install -r req.txt
```

**Core packages:**

```
# CLIP pipeline
lightning==2.5.0
transformers==4.50.0
peft==0.14.0
scikit-learn==1.6.1
tqdm==4.67.1
matplotlib==3.10.0
seaborn==0.13.2
pydantic==2.9.2
fire==0.7.0
wandb==0.19.4

# EAR pipeline (additional)
dlib>=19.24.0
mediapipe>=0.10.0
scipy>=1.12.0
```

> **GPU:** Minimum 16 GB VRAM for ViT-L/14 training. ViT-B/16 and ViT-B/32 run on 8 GB. The EAR pipeline runs entirely on CPU — no GPU needed.

> **dlib:** Requires CMake and a C++ compiler.  
> Ubuntu: `sudo apt-get install cmake build-essential`  
> macOS: `brew install cmake`  
> The 68-point landmark model (`shape_predictor_68_face_landmarks.dat`) downloads automatically on first run, or place it manually in the project root.

---

## 6. Dataset Preparation

> Both pipelines share the same face crop extraction step. Run it once and both pipelines will use the output.

### 6.1 Download FaceForensics++

Request access at [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics), then download:

```bash
python download.py /path/to/output \
  --dataset all \
  --compression c23 \
  --type videos \
  --server EU
```

Downloads: `original` (YouTube real), `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`.

### 6.2 Extract Face Crops from FF++

Edit the top of `perpare_dataset.py`:

```python
VIDEO_ROOT  = 'path/to/FF/videos'
OUTPUT_ROOT = 'datasets/FF'
```

```bash
python perpare_dataset.py
```

Extracts one face crop every 10 frames at 224×224 using MTCNN (thresholds `[0.5, 0.6, 0.6]` to handle c23 compression). Saves up to 10 crops per video.

```
datasets/FF/
├── DF/000_003/000.png  001.png  ...
├── F2F/000_003/000.png ...
├── FS/000_003/000.png  ...
├── NT/000_003/000.png  ...
└── real/000/000.png    ...
```

### 6.3 Generate FF++ Manifests

```bash
bash scripts/prepare_FF.sh
```

Creates `config/datasets/FF/test/{DF,F2F,FS,NT,real}.txt` — one image path per line.

### 6.4 Download and Prepare Celeb-DF-v2

Download CDFv2 from [yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) and place under:

```
archive/
├── Celeb-real/
├── Celeb-synthesis/
└── YouTube-real/
```

Extract face crops:

```bash
# Edit perpare_dataset.py:
# VIDEO_ROOT = 'archive'
# OUTPUT_ROOT = 'datasets/CDF2'
python perpare_dataset.py
```

Generate manifests:

```bash
bash data_conversion.sh
```

---

## 7. CLIP Pipeline — Training and Evaluation

### 7.1 Train — Primary Experiment (ViT-L/14, LN-Tuning)

Default config in `get_train_config()` inside `train.py`:

| Setting | Value |
|---|---|
| Backbone | `openai/clip-vit-large-patch14` |
| PEFT | LN-Tuning (LoRA disabled) |
| Head | LinearNorm |
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 8e-5 → 5e-5 (cosine annealing) |
| Slerp augmentation | Enabled over `[0.0, 1.0]` |
| Mixed precision | bfloat16 |
| Random seed | 42 |

```bash
python train.py --train
```

### 7.2 Train Other Backbones

Edit `get_train_config()` in `train.py`:

```python
config.backbone = Backbone.CLIP_B_16   # ViT-B/16
config.backbone = Backbone.CLIP_B_32   # ViT-B/32
```

```bash
python train.py --train
```

### 7.3 Debug Run (verify setup before full training)

```bash
python train.py --train --debug
```

Runs 12 training + 12 validation batches.

### 7.4 Test / Evaluate

Update `get_test_config()` in `train.py`:

```python
config_path = "runs/train/your-run-name/hparams.yaml"
```

```bash
python train.py --test
```

**Outputs saved to the run directory:**

```
test_predictions.csv
test/frame_metrics/roc.png
test/frame_metrics/confusion_matrix.png
test/frame_metrics/f1_curve.png
test/video_metrics/roc.png
test/video_metrics/confusion_matrix.png
metrics.csv
```

---

## 8. EAR Pipeline — Original Contribution

The EAR pipeline is implemented in `ear_baseline.py` and `src/ear/`. It operates **completely independently** of the CLIP pipeline and uses the same face crops from Step 6.

### 8.1 How It Works

For each video (ordered sequence of face crop PNGs):

1. Apply **68-point facial landmarks** (dlib) to each frame
2. Locate the six periorbital keypoints per eye (landmarks 36–41 left, 42–47 right)
3. Compute **Eye Aspect Ratio** per frame:

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

4. Average left and right EAR
5. Detect blink events by threshold crossing (threshold = 0.25)
6. Build a **9-dimensional feature vector** per video:

| # | Feature | Description |
|---|---|---|
| 0 | `mean_ear` | Mean EAR across frames |
| 1 | `std_ear` | Standard deviation of EAR |
| 2 | `min_ear` | Minimum EAR (deepest blink point) |
| 3 | `blink_count` | Number of detected blink events |
| 4 | `mean_blink_duration` | Mean blink duration in frames |
| 5 | `mean_inter_blink_interval` | Mean gap between blinks in frames |
| 6 | `fft_energy` | Spectral energy of EAR time series |
| 7 | `kurtosis` | Kurtosis of EAR distribution |
| 8 | `skewness` | Skewness of EAR distribution |

7. Fit **StandardScaler + LogisticRegression** on FF++ features
8. Evaluate on FF++ (in-distribution) and CDFv2 (cross-dataset, no re-fitting)

### 8.2 Run Step by Step

**Extract features from FF++:**

```bash
python ear_baseline.py --extract \
  --dataset_root datasets/FF \
  --output_dir ear_features/FF
```

**Train classifier on FF++:**

```bash
python ear_baseline.py --train \
  --features_dir ear_features/FF \
  --output_dir ear_models/
```

**Evaluate on FF++ (in-distribution):**

```bash
python ear_baseline.py --evaluate \
  --features_dir ear_features/FF \
  --model_path ear_models/ear_classifier.pkl \
  --output_dir results/ear_FF
```

**Extract CDFv2 features and cross-dataset evaluate:**

```bash
python ear_baseline.py --extract \
  --dataset_root datasets/CDF2 \
  --output_dir ear_features/CDFv2

python ear_baseline.py --evaluate \
  --features_dir ear_features/CDFv2 \
  --model_path ear_models/ear_classifier.pkl \
  --output_dir results/ear_CDFv2 \
  --cross_dataset
```

### 8.3 Run Everything in One Command

```bash
python ear_baseline.py --run_all \
  --ff_root datasets/FF \
  --cdf_root datasets/CDF2 \
  --output_dir results/ear \
  --seed 42
```

### 8.4 EAR Results

| Metric | FF++ (in-distribution) | CDFv2 (cross-dataset) |
|---|---|---|
| AUC-ROC | 0.4871 | 0.7866 |
| Best threshold (τ) | 0.165 | 0.200 |
| Balanced Accuracy | 0.5076 | 0.7375 |
| Accuracy | 0.7417 | 0.7417 |
| Precision | 0.8142 | 0.5918 |
| Recall (Fake) | 0.0623 | 0.7250 |
| F1 Score | 0.1157 | 0.6517 |
| Real → TN \| FP | 952 \| 47 | 60 \| 20 |
| Fake → FN \| TP | 3101 \| 206 | 11 \| 29 |

The AUC of 0.4871 on FF++ (marginally below chance) confirms that standalone EAR features do not discriminate between real and fake across the four FF++ manipulation categories at c23 compression — consistent with Jung et al. (2020). The improved CDFv2 result (0.7866) reflects the smaller, more balanced test distribution rather than genuine generalisation.

---

## 9. Reproducing Published Results

### CLIP Runs

| Run name | Backbone | Config file |
|---|---|---|
| `clip_b16_26_apr` | ViT-B/16 | `runs/train/clip_b16_26_apr/hparams.yaml` |
| `clip_b14_27_apr` | ViT-B/32 | `runs/train/clip_b14_27_apr/hparams.yaml` |
| `clip_d14_1_05` | ViT-L/14 | `runs/train/clip_d14_1_05/hparams.yaml` |

All runs: seed 42 · LN-Tuning · LoRA disabled · CDFv2 val+test · no target-domain fine-tuning.

```bash
# Reproduce ViT-L/14
python train.py --train --backbone openai/clip-vit-large-patch14 --run_name clip_d14_1_05

# Reproduce ViT-B/16
python train.py --train --backbone openai/clip-vit-base-patch16 --run_name clip_b16_26_apr

# Reproduce ViT-B/32
python train.py --train --backbone openai/clip-vit-base-patch32 --run_name clip_b14_27_apr
```

### EAR Run

```bash
python ear_baseline.py --run_all \
  --ff_root datasets/FF \
  --cdf_root datasets/CDF2 \
  --output_dir results/ear \
  --seed 42
```

---

## 10. Full Results Table

| Metric | ViT-B/32 | ViT-B/16 | ViT-L/14 | EAR-only |
|---|---|---|---|---|
| Frame AUC-ROC | 0.9470 | 0.9629 | **0.9753** | 0.4871 † |
| Frame Balanced Acc. | 0.7711 | 0.8280 | **0.8469** | 0.5076 † |
| Frame F1 | 0.7029 | 0.7794 | **0.8019** | 0.1157 † |
| Frame EER | 0.1284 | 0.0959 | **0.0839** | — |
| Video AUC-ROC | **0.9991** | 0.9962 | 0.9984 | 0.7866 |
| Video mAP | **0.9989** | 0.9956 | 0.9982 | — |
| Video Balanced Acc. | 0.7750 | 0.8688 | **0.8938** | 0.7375 |
| Video F1 | 0.6997 | 0.8205 | **0.8529** | 0.6517 |
| Video EER | **0.0250** | 0.0500 | **0.0250** | — |
| Fake Recall (Video) | 100% | 100% | 100% | 72.50% |
| Real FP Rate (Video) | 45.00% | 26.25% | **21.25%** | 25.00% |
| Test Loss (CE) | 0.7051 | 0.5976 | **0.5574** | — |

† EAR frame-level figures evaluated on FF++ (in-distribution). All CLIP figures evaluated on CDFv2 (cross-dataset). No target-domain fine-tuning at any stage.

---

## 11. Configuration Reference

All hyperparameters are managed by the Pydantic `Config` class in `src/config.py`. Every run serialises its full configuration to `hparams.yaml` alongside the checkpoint for exact reproducibility.

```python
config.backbone                    # CLIP model ID (e.g. 'openai/clip-vit-large-patch14')
config.peft.enabled                # Enable/disable PEFT
config.peft.ln_tuning.enabled      # LN-Tuning (True in all reported runs)
config.peft.lora.enabled           # LoRA (False in all reported runs)
config.head                        # 'linear' or 'LinearNorm'
config.lr                          # Peak learning rate  [default: 8e-5]
config.min_lr                      # Cosine floor LR     [default: 5e-5]
config.max_epochs                  # Training epochs     [default: 30]
config.batch_size                  # Batch size          [default: 8]
config.slerp_feature_augmentation  # Slerp augmentation  [default: True]
config.seed                        # Random seed         [default: 42]
```

---

## 12. Dataset Licences

Both datasets are available for **non-commercial academic research only**:

- **FaceForensics++** — [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) — requires acceptance of the FaceForensics Terms of Service before downloading.
- **Celeb-DF-v2** — [yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) — available under academic use licence.

---

## 13. Citation

If you use this code or the EAR baseline in your research, please cite:

```bibtex
@mastersthesis{shah2025deepfake,
  author  = {Shah, Kabir},
  title   = {Explainable Deepfake Detection Using CLIP and Eye-Blink Dynamics},
  school  = {GISMA University of Applied Sciences},
  year    = {2025},
  note    = {MSc Data Science and Artificial Intelligence, Module M599},
  url     = {https://github.com/Kdshah1512/DeepFake-detection-project}
}
```

---

## 14. Acknowledgements

This work builds upon the open-source CLIP deepfake detection pipeline by [Yermakov (2025)](https://github.com). The EAR feature extraction follows the landmark-based approach of Soukupová & Čech (2016) and the blink-based detection methodology of Li et al. (2018) and Jung et al. (2020).

**Key dependencies:** PyTorch · PyTorch Lightning · Hugging Face Transformers · PEFT · facenet-pytorch · dlib · scikit-learn · OpenCV · NumPy · scipy · matplotlib

