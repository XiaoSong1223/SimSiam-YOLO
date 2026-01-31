# 🚦 SimSiam-YOLOv8: Self-Supervised Learning for Traffic Sign Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

> **基于SimSiam自监督学习与YOLOv8融合的道路交通标志检测研究**
>
> A Study on Road Traffic Sign Detection Based on SimSiam Self-Supervised Learning and YOLOv8 Fusion

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Feature--Level_SimSiam-green" alt="Framework">
  <img src="https://img.shields.io/badge/Backbone-YOLOv8n-orange" alt="Backbone">
  <img src="https://img.shields.io/badge/Dataset-TT--100K-purple" alt="Dataset">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Self-Supervised Pre-training](#1-self-supervised-pre-training)
  - [Weight Conversion](#2-weight-conversion)
  - [Supervised Fine-tuning](#3-supervised-fine-tuning)
- [Experimental Results](#-experimental-results)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## 💻 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for NVIDIA GPU) or MPS (for Apple Silicon)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/simsiam-yolov8.git
cd simsiam-yolov8

# Create virtual environment
conda create -n simsiam-yolo python=3.10
conda activate simsiam-yolo

# Install PyTorch (choose one based on your hardware)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS)
pip install torch torchvision torchaudio

# Install dependencies
pip install ultralytics opencv-python matplotlib pyyaml tqdm
```

---

## 📁 Project Structure

```
ThesisII/
├── 📂 models_1/
│   ├── 📂 my_experiment/           # Core implementation
│   │   ├── simsiam_yolo.py        # SimSiamYOLO model
│   │   ├── yolo_encoder.py        # YOLOv8 backbone encoder
│   │   ├── train.py               # Pre-training script
│   │   ├── train_finetune.py      # Fine-tuning script
│   │   ├── convert_weights.py     # Weight conversion tool
│   │   └── prepare_data_split.py  # Data preparation
│   │
│   ├── 📂 simsiam-main/           # Original SimSiam reference
│   ├── 📂 ultralytics/            # YOLOv8 framework
│   ├── 📂 checkpoints/            # Pre-training checkpoints
│   └── 📂 runs/                   # Training outputs
│
├── 📂 dataset/                    # Datasets
│   ├── tt100k_2021/              # Full TT-100K dataset
│   └── TT100K_Subsets/           # Sampled subsets
│
├── 📂 thesis/                     # Thesis documents
│   ├── thesis_main.md
│   ├── chapter1_introduction.md
│   └── ...
│
└── README.md
```

---

## 📖 Usage

### 1. Self-Supervised Pre-training

Train SimSiamYOLO on unlabeled road scene images:

```bash
cd models_1

# Basic training
python -m my_experiment.train \
    /path/to/unlabeled/images \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.05 \
    --imgsz 640 \
    --save-dir ./checkpoints

# With pretrained YOLOv8 weights
python -m my_experiment.train \
    /path/to/unlabeled/images \
    --weights yolov8n.pt \
    --epochs 100
```

### 2. Weight Conversion

Convert SimSiamYOLO checkpoint to YOLOv8 format:

```bash
python -m my_experiment.convert_weights \
    ./checkpoints/checkpoint_0099.pth.tar \
    --cfg yolov8n.yaml \
    --output yolov8_simsiam_pretrained.pt
```

### 3. Supervised Fine-tuning

Fine-tune on labeled traffic sign dataset:

```bash
# Mode A: Our method (SimSiam pretrained + Progressive Unfreezing)
python -m my_experiment.train_finetune \
    --mode ours \
    --data /path/to/data.yaml \
    --epochs 100 \
    --freeze 10

# Mode B: Baseline (ImageNet pretrained)
python -m my_experiment.train_finetune \
    --mode baseline \
    --data /path/to/data.yaml \
    --epochs 100

# Mode C: From scratch
python -m my_experiment.train_finetune \
    --mode scratch \
    --data /path/to/data.yaml \
    --epochs 100
```

<p align="center">
  <b>⭐ Star this repo if you find it helpful! ⭐</b>
</p>

<p align="center">
  Made with ❤️ at Xiamen University Malaysia
</p>

