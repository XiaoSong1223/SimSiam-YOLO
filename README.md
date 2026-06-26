# SimSiam-YOLOv8: Self-Supervised Traffic Sign Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![SimSiam](https://img.shields.io/badge/SSL-SimSiam-green.svg)](https://github.com/facebookresearch/simsiam)

This repository contains the implementation for **Traffic Sign Detection Using SimSiam-Enhanced YOLOv8 With Limited Labeled Data**.

The project explores a data-efficient detection pipeline for traffic signs: first learn visual representations from unlabeled road-scene images using a feature-level SimSiam objective, then transfer the learned YOLOv8 backbone weights into supervised fine-tuning on limited labeled TT-100K subsets.

## Highlights

- Feature-level SimSiam pre-training on YOLOv8 backbone feature maps.
- Multi-scale dense contrastive learning over P3, P4, and P5 features.
- Weight conversion utility for transferring self-supervised backbone weights into YOLOv8 detection models.
- Fine-tuning workflow for three comparison modes: SimSiam pre-trained, ImageNet baseline, and training from scratch.
- Clean research-oriented repository layout with project code separated from vendored reference implementations.

## Repository structure

```text
SimSiam-YOLO/
├── README.md
├── pyproject.toml
├── src/
│   └── simsiam_yolo/
│       ├── models/
│       │   ├── simsiam_yolo.py
│       │   └── yolo_encoder.py
│       ├── training/
│       │   ├── pretrain.py
│       │   └── finetune.py
│       ├── data/
│       │   └── prepare_data_split.py
│       └── tools/
│           └── convert_weights.py
├── external/
│   ├── ultralytics/      # Vendored YOLOv8 framework snapshot
│   └── simsiam/          # Original SimSiam reference implementation
└── docs/
    └── thesis/           # Body-only thesis PDF
```

Generated training outputs, checkpoints, datasets, and converted weights should stay outside source control. Suggested local folders are `data/`, `artifacts/weights/`, `checkpoints/`, and `runs/`; these are ignored by `.gitignore`.

## Installation

```bash
git clone https://github.com/XiaoSong1223/SimSiam-YOLO.git
cd SimSiam-YOLO

conda create -n simsiam-yolo python=3.10
conda activate simsiam-yolo

# Install PyTorch for your hardware first, for example:
pip install torch torchvision torchaudio

# Install this project in editable mode
pip install -e .
```

The code first tries to use the vendored `external/ultralytics/` snapshot. If that folder is unavailable, it falls back to the installed `ultralytics` package.

## Data preparation

Prepare a limited-label TT-100K split:

```bash
python -m simsiam_yolo.data.prepare_data_split \
  --train-images data/tt100k_2021/train/images \
  --train-labels data/tt100k_2021/train/labels \
  --original-yaml data/tt100k_2021/data.yaml \
  --output-dir data/TT100K_Subsets/train_10percent \
  --percentage 0.10 \
  --seed 42
```

## Workflow

### 1. Self-supervised pre-training

```bash
python -m simsiam_yolo.training.pretrain \
  data/unlabeled/images \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.05 \
  --imgsz 640 \
  --save-dir checkpoints/simsiam_yolo
```

Optional YOLOv8 initialization:

```bash
python -m simsiam_yolo.training.pretrain \
  data/unlabeled/images \
  --weights yolov8n.pt \
  --epochs 100
```

### 2. Convert SimSiam weights to YOLOv8 format

```bash
python -m simsiam_yolo.tools.convert_weights \
  checkpoints/simsiam_yolo/checkpoint_0099.pth.tar \
  --cfg yolov8n.yaml \
  --output artifacts/weights/yolov8_simsiam_pretrained.pt
```

### 3. Supervised fine-tuning

SimSiam-enhanced YOLOv8:

```bash
python -m simsiam_yolo.training.finetune \
  --mode ours \
  --data data/TT100K_Subsets/train_10percent/TT100K_10percent.yaml \
  --pretrained-weights artifacts/weights/yolov8_simsiam_pretrained.pt \
  --epochs 100 \
  --freeze 10
```

ImageNet/standard YOLOv8 baseline:

```bash
python -m simsiam_yolo.training.finetune \
  --mode baseline \
  --data data/TT100K_Subsets/train_10percent/TT100K_10percent.yaml \
  --epochs 100
```

Training from scratch:

```bash
python -m simsiam_yolo.training.finetune \
  --mode scratch \
  --data data/TT100K_Subsets/train_10percent/TT100K_10percent.yaml \
  --epochs 100
```

## Thesis

The body-only thesis PDF is available at:

- [`docs/thesis/Traffic_Sign_Detection_Using_SimSiam-Enhanced_YOLOv8_With_Limited_Labeled_Data.pdf`](docs/thesis/Traffic_Sign_Detection_Using_SimSiam-Enhanced_YOLOv8_With_Limited_Labeled_Data.pdf)

## Acknowledgements

This work builds on:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam)
- TT-100K traffic sign detection dataset

Please review the licenses in `external/ultralytics/` and `external/simsiam/` when reusing or redistributing this project.
