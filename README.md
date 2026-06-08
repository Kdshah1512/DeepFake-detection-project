# Deepfake Detection Using CLIP Vision Transformers and Eye-Blink Analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.5.0-purple.svg)](https://lightning.ai/)
[![License: Academic](https://img.shields.io/badge/License-Academic_Use_Only-green.svg)]()

**MSc Data Science, Artificial Intelligence and Business — Master's Dissertation**  
GISMA University of Applied Sciences, 2025  
Module: M599 — Master Thesis  
Author: **Kirtikumar Shah**

---

## Overview

This repository contains the full implementation of a parameter-efficient deepfake detection pipeline built on OpenAI's CLIP Vision Transformer encoder. The system is trained on the **FaceForensics++** dataset and cross-evaluated on **Celeb-DF-v2 (CDFv2)** without any target-domain fine-tuning.

Three CLIP backbone configurations are evaluated:

| Backbone | Embedding Dim | Patch Size | Video AUC-ROC (CDFv2) |
|---|---|---|---|
| ViT-B/32 | 768 | 32×32 | 0.9991 |
| ViT-B/16 | 768 | 16×16 | 0.9962 |
| **ViT-L/14 ★** | **1024** | **14×14** | **0.9984** |

★ Primary model — best frame-level AUC-ROC (0.9753) and video-level balanced accuracy (0.8938).

All three configurations achieve **100% fake recall** at the video level on CDFv2.

---

## Research Contributions

This dissertation makes the following contributions:

1. **Parameter-efficient CLIP adaptation** — Layer Normalisation Tuning (LN-Tuning) updates fewer than 0.01% of backbone parameters while achieving competitive cross-dataset generalisation.
2. **Three-way backbone comparison** — ViT-B/16, ViT-B/32, and ViT-L/14 evaluated under identical training and evaluation conditions using a strict FaceForensics++ → Celeb-DF-v2 protocol.
3. **EAR physiological baseline** — Implementation of a standalone Eye Aspect Ratio (EAR) based physiological baseline using facial landmark analysis and blink dynamics, providing an interpretable behavioural reference model for comparison with CLIP visual representations.
4. **Cross-dataset evaluation** — Empirical comparison between CLIP visual representations and EAR-based physiological features under the same cross-dataset evaluation protocol, with no target-domain fine-tuning.
5. **Reproducible evaluation framework** — Full pipeline supporting AUC-ROC, mAP, EER, Precision, Recall, F1 Score, Balanced Accuracy, ROC curves, Precision–Recall curves, and confusion matrix analysis.

---

## EAR Physiological Baseline

A standalone Eye Aspect Ratio (EAR) based physiological baseline was implemented to investigate whether blink dynamics provide useful information for deepfake detection.

The pipeline uses dlib 68-point facial landmark detection to locate eye regions and compute EAR values per frame. Video-level EAR measurements are then evaluated as an alternative detection signal and compared directly with CLIP-based visual representations under the same cross-dataset evaluation protocol.

The EAR baseline serves as an interpretable physiological reference model, enabling a direct comparison between behavioural eye-blink patterns and deep visual features extracted by CLIP Vision Transformers.

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

---

## Repository Structure

```
.
├── train.py                    # Main entry point — training and testing
├── ear_only_classifiers.py     # EAR classifier pipeline
├── evaluate_ear_only.py        # EAR evaluation pipeline
├── prepare_dataset.py          # Face extraction pipeline (MTCNN)
├── download.py                 # FaceForensics++ official downloader
├── data_conversion.sh          # CDFv2 manifest generation script
├── req.txt                     # Python dependencies
│
├── src/
│   ├── config.py               # Pydantic-based configuration system
│   ├── loss.py                 # Cross-entropy loss
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
│   │   └── dfdet.py            # DeepfakeDetectionModel (Lightning module)
│   │
│   ├── dataset/
│   │   ├── base.py             # BaseDataset and BaseDataModule
│   │   └── deepfake.py         # DeepfakeDataset and DeepfakeDataModule
│   │
│   └── losses/
│       └── unifalign.py        # Alignment and uniformity loss functions
│
└── config/
    └── datasets/
        ├── FF/test/            # FaceForensics++ file manifests
        │   ├── DF.txt
        │   ├── F2F.txt
        │   ├── FS.txt
        │   ├── NT.txt
        │   └── real.txt
        └── CDFv2/test/         # Celeb-DF-v2 file manifests
            ├── Celeb-synthesis.txt
            ├── Celeb-real.txt
            └── YouTube-real.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kdshah1512/DeepFake-detection-project.git
cd DeepFake-detection-project
```

### 2. Create Environment

```bash
conda create -n dfdet python=3.12
conda activate dfdet
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```

Core dependencies:

```
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
dlib
```

> **GPU requirement:** Minimum 16 GB VRAM for ViT-L/14 training. ViT-B/16 and ViT-B/32 run on 8 GB VRAM. Inference via a saved checkpoint can run on 6 GB VRAM.

---

## Dataset Preparation

### Step 1 — Download FaceForensics++

Request access and download using the official script:

```bash
python download.py /path/to/output \
  --dataset all \
  --compression c23 \
  --type videos \
  --server EU
```

This downloads all five categories: `original` (YouTube real), `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`.

### Step 2 — Extract Face Crops from FF++

Edit the paths in `prepare_dataset.py`:

```python
VIDEO_ROOT = 'path/to/FF/videos'
OUTPUT_ROOT = 'datasets/FF'
```

Then run:

```bash
python prepare_dataset.py
```

This extracts one face crop every 10 frames at 224×224 resolution using MTCNN. Output structure:

```
datasets/FF/
├── DF/000_003/000.png ...
├── F2F/000_003/000.png ...
├── FS/000_003/000.png ...
├── NT/000_003/000.png ...
└── real/000/000.png ...
```

### Step 3 — Generate File Manifests for FF++

```bash
bash scripts/prepare_FF.sh
```

### Step 4 — Prepare Celeb-DF-v2

Download CDFv2 from the [official source](https://github.com/yuezunli/celeb-deepfakeforensics) and place under:

```
archive/
├── Celeb-real/
├── Celeb-synthesis/
└── YouTube-real/
```

Extract face crops and generate manifests:

```bash
python prepare_dataset.py   # ensure VIDEO_ROOT = 'archive', OUTPUT_ROOT = 'datasets/CDF2'
bash data_conversion.sh
```

---

## Training

### Primary Experiment — ViT-L/14 with LN-Tuning

```bash
python train.py --train
```

Default configuration (set in `get_train_config()` inside `train.py`):

- Backbone: `openai/clip-vit-large-patch14`
- PEFT: LN-Tuning enabled, LoRA disabled
- Head: LinearNorm
- Epochs: 30
- Batch size: 8
- Learning rate: 8e-5 → 5e-5 (cosine)
- Slerp augmentation: enabled
- Seed: 42

### Other Backbones

Edit `get_train_config()` in `train.py`:

```python
# For ViT-B/16
config.backbone = Backbone.CLIP_B_16

# For ViT-B/32
config.backbone = Backbone.CLIP_B_32
```

### Debug Run

```bash
python train.py --train --debug
```

Runs 12 training and 12 validation batches to verify the pipeline before a full training run.

---

## Testing / Evaluation

### CLIP Models

```bash
python train.py --test
```

Update `get_test_config()` in `train.py` to point to your saved checkpoint:

```python
config_path = "runs/train/your-run-name/hparams.yaml"
```

The test script loads the best checkpoint (`best_mAP.ckpt`) and saves:

- `test_predictions.csv` — per-frame probabilities
- `test/frame_metrics/*.png` — ROC, PRC, F1, confusion matrix (frame level)
- `test/video_metrics/*.png` — ROC, PRC, F1, confusion matrix (video level)
- `metrics.csv` — training and validation metrics history

### EAR Baseline

```bash
python evaluate_ear_only.py
```

---

## Reproducing Published Results

| Run Name | Backbone | hparams |
|---|---|---|
| `clip_b16_26_apr` | ViT-B/16 | `runs/train/clip_b16_26_apr/hparams.yaml` |
| `clip_b14_27_apr` | ViT-B/32 | `runs/train/clip_b14_27_apr/hparams.yaml` |
| `clip_d14_1_05` | ViT-L/14 | `runs/train/clip_d14_1_05/hparams.yaml` |

All runs use random seed `42`, LN-Tuning, and the same FF++ → CDFv2 evaluation protocol with no target-domain fine-tuning.

---

## Key Results (CDFv2 Cross-Dataset Evaluation)

| Metric | ViT-B/32 | ViT-B/16 | ViT-L/14 |
|---|---|---|---|
| Frame AUC-ROC | 0.9470 | 0.9629 | **0.9753** |
| Frame mAP | 0.9402 | 0.9592 | **0.9736** |
| Frame Balanced Acc. | 0.7711 | 0.8280 | **0.8469** |
| Frame F1 | 0.7029 | 0.7794 | **0.8019** |
| Frame EER | 0.1284 | 0.0959 | **0.0839** |
| Video AUC-ROC | **0.9991** | 0.9962 | 0.9984 |
| Video Balanced Acc. | 0.7750 | 0.8688 | **0.8938** |
| Video F1 | 0.6997 | 0.8205 | **0.8529** |
| Video EER | **0.0250** | 0.0500 | **0.0250** |
| Fake Recall (Video) | 100% | 100% | 100% |
| Real FP Rate (Video) | 45.00% | 26.25% | **21.25%** |

All models trained on FaceForensics++ (c23), evaluated on CDFv2 with no target-domain fine-tuning.

**EAR Baseline:** AUC-ROC 0.7866 on CDFv2. CLIP Vision Transformers substantially outperform the EAR-based physiological baseline in cross-dataset evaluation.

---

## Evaluation Metrics

The framework supports:

- AUC-ROC (frame-level and video-level)
- Mean Average Precision (mAP)
- Equal Error Rate (EER)
- Precision, Recall, F1 Score
- Balanced Accuracy
- ROC Curves
- Precision–Recall Curves
- Confusion Matrix Analysis

---

## Configuration Reference

All hyperparameters are managed by the Pydantic-based `Config` class in `src/config.py`. Every training run saves its configuration to `hparams.yaml` alongside the checkpoint for exact reproducibility.

Key options:

```python
config.backbone                    # CLIP model identifier
config.peft.enabled                # Enable/disable PEFT
config.peft.ln_tuning.enabled      # Enable LN-Tuning
config.peft.lora.enabled           # Enable LoRA (disabled in all reported runs)
config.head                        # 'linear' or 'LinearNorm'
config.lr                          # Peak learning rate
config.max_epochs                  # Training epochs
config.slerp_feature_augmentation  # Slerp latent augmentation
config.seed                        # Global random seed (default: 42)
```

---

## Dataset Licences

Both datasets are available for **non-commercial academic research only**:

- **FaceForensics++** — [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) — requires agreement with the FaceForensics TOS before downloading.
- **Celeb-DF-v2** — [yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) — available under academic use licence.

---

## Author

**Kirtikumar Shah**  
MSc Data Science, Artificial Intelligence and Business  
GISMA University of Applied Sciences  
2025

---

## Citation

```bibtex
@mastersthesis{shah2025deepfake,
  author  = {Kirtikumar Shah},
  title   = {Deepfake Detection Using CLIP Vision Transformers and Eye-Blink Analysis},
  school  = {GISMA University of Applied Sciences},
  year    = {2025},
  note    = {MSc Data Science, Artificial Intelligence and Business, Module M599},
  url     = {https://github.com/Kdshah1512/DeepFake-detection-project}
}
```
