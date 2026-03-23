#!/usr/bin/env python
"""
Fine-tuning experiment script for TT-100K dataset with limited labeled data.

This script supports three training modes:
- Mode A (Ours): Use SimSiam pre-trained weights with backbone freezing strategy
- Mode B (Baseline): Use ImageNet pre-trained weights (standard transfer learning)
- Mode C (Scratch): Train from scratch without any pre-trained weights

Usage:
    # Mode A: Train with SimSiam pre-trained weights (freeze backbone for first 10 epochs)
    python -m my_experiment.train_finetune --mode ours --epochs 100 --freeze 10
    
    # Mode B: Train with ImageNet pre-trained weights (no freezing)
    python -m my_experiment.train_finetune --mode baseline --epochs 100
    
    # Mode C: Train from scratch (no pre-trained weights, no freezing)
    python -m my_experiment.train_finetune --mode scratch --epochs 100
    
    # Custom data path
    python -m my_experiment.train_finetune --mode ours --data ../dataset/tt100k_20pct/data_20pct.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add ultralytics to path
_ultralytics_path = Path(__file__).parent.parent / "ultralytics"
_ultralytics_path = _ultralytics_path.resolve()  # Use absolute path
if _ultralytics_path.exists():
    if str(_ultralytics_path) not in sys.path:
        sys.path.insert(0, str(_ultralytics_path))
else:
    print(f"Warning: ultralytics path not found: {_ultralytics_path}")
    print("  Trying to import from system path...")

try:
    from ultralytics import YOLO
except ImportError as e:
    error_msg = str(e)
    if "torch" in error_msg.lower() or "No module named 'torch'" in error_msg:
        print("Error: PyTorch (torch) is not installed.")
        print("  Please install PyTorch first:")
        print("    pip install torch torchvision")
        print("  Or for Apple Silicon:")
        print("    pip install torch torchvision")
    else:
        print(f"Error: ultralytics not found. Please install it or add to path.")
        print(f"  Tried path: {_ultralytics_path}")
        print(f"  Path exists: {_ultralytics_path.exists()}")
        print(f"  Import error: {e}")
    sys.exit(1)

try:
    import torch
except ImportError:
    torch = None


def get_model_path(mode: str) -> str:
    """
    Get model path based on training mode.
    
    Args:
        mode: 'ours', 'baseline', or 'scratch'
    
    Returns:
        Path to model weights file or config file
    """
    if mode == 'ours':
        # SimSiam pre-trained weights
        model_path = Path(__file__).parent / "yolov8_simsiam_100pretrained.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"SimSiam weights not found at {model_path}. "
                "Please run convert_weights.py first."
            )
        return str(model_path.absolute())
    
    elif mode == 'baseline':
        # Standard ImageNet pre-trained weights
        return 'yolov8n.pt'  # Will be downloaded automatically if not exists
    
    elif mode == 'scratch':
        # Train from scratch using YAML config (no pre-trained weights)
        return 'yolov8n.yaml'
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'ours', 'baseline', or 'scratch'.")


def get_auto_device() -> str:
    """
    Automatically select device with priority: CUDA > MPS > CPU.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch is None:
        return 'cpu'
    
    # Priority 1: CUDA
    if torch.cuda.is_available():
        return 'cuda'
    
    # Priority 2: MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    
    # Priority 3: CPU (fallback)
    return 'cpu'


def unfreeze_backbone(model):
    """
    Unfreeze all backbone layers in the model.
    
    Args:
        model: YOLO model instance
    """
    print("Unfreezing backbone layers...")
    unfrozen_count = 0
    for name, param in model.model.named_parameters():
        if any(f'model.{i}.' in name for i in range(10)):  # Backbone layers 0-9
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
    print(f"  Unfrozen {unfrozen_count} backbone parameters")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tuning experiment for TT-100K with limited labeled data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode A: Train with SimSiam weights (freeze backbone for 10 epochs)
  python -m my_experiment.train_finetune --mode ours --epochs 100 --freeze 10
  
  # Mode B: Train with ImageNet weights (no freezing)
  python -m my_experiment.train_finetune --mode baseline --epochs 100
  
  # Mode C: Train from scratch (no pre-trained weights)
  python -m my_experiment.train_finetune --mode scratch --epochs 100
  
  # Use 20% dataset
  python -m my_experiment.train_finetune --mode ours --data ../dataset/tt100k_20pct/data_20pct.yaml --freeze 10
  
  # Custom batch size and image size
  python -m my_experiment.train_finetune --mode ours --batch 8 --imgsz 512 --freeze 10
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['ours', 'baseline', 'scratch'],
        required=True,
        help='Training mode: "ours" (SimSiam), "baseline" (ImageNet), or "scratch" (no pretrain)'
    )
    
    # Data configuration
    parser.add_argument(
        '--data',
        type=str,
        default='/Users/xiaosongxiaosong/Course/ThesisII/dataset/TT100K_Subsets/train_10percent/TT100K_10percent.yaml',
        help='Path to dataset YAML file (default: /Users/xiaosongxiaosong/Course/ThesisII/dataset/TT100K_Subsets/train_10percent/TT100K_10percent.yaml)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        help='Optimizer type: auto, SGD, AdamW (default: auto)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='Momentum for SGD (default: 0.937)'
    )
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=0.0005,
        help='Weight decay (default: 5e-4)'
    )
    
    # Freezing strategy (only for Mode A)
    parser.add_argument(
        '--freeze',
        type=int,
        default=10,
        help='Number of epochs to freeze backbone (default: 10, only for Mode A)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (auto-detect if not specified: CUDA > MPS > CPU)'
    )
    
    # Output configuration
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not specified)'
    )
    
    # Other options
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--pin-memory',
        dest='pin_memory',
        action='store_true',
        help='Enable DataLoader pin_memory (default: False)'
    )
    parser.add_argument(
        '--no-pin-memory',
        dest='pin_memory',
        action='store_false',
        help='Disable DataLoader pin_memory'
    )
    parser.set_defaults(pin_memory=False)
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save training checkpoints'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Auto-select device if not specified
    if args.device is None:
        args.device = get_auto_device()
        print(f"Auto-selected device: {args.device}")
    
    # Auto-generate experiment name
    if args.name is None:
        dataset_name = Path(args.data).parent.name
        args.name = f"finetune_{dataset_name}_{args.mode}"
    
    print("=" * 60)
    print("Fine-tuning Experiment")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"  - Ours: SimSiam pre-trained weights")
    print(f"  - Baseline: ImageNet pre-trained weights")
    print(f"  - Scratch: Train from scratch (no pretrain)")
    print()
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print()
    
    if args.mode == 'ours':
        print(f"Freezing strategy: Backbone frozen for first {args.freeze} epochs")
        print()
    elif args.mode == 'scratch':
        print("Training from scratch: All parameters randomly initialized")
        print()
    
    # Get model path
    try:
        model_path = get_model_path(args.mode)
        print(f"Loading model from: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create model
    model = YOLO(model_path)
    
    if args.mode == 'ours' and args.freeze > 0:
        # For Mode A with freezing, we need a two-stage training approach
        print("\n" + "=" * 60)
        print("Stage 1: Training with frozen backbone")
        print(f"Freezing backbone layers (0-9) for first {args.freeze} epochs...")
        print("=" * 60)
        
        # Stage 1: Train with frozen backbone (freeze=10 means freeze layers 0-9)
        results_stage1 = model.train(
            data=args.data,
            epochs=args.freeze,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=f"{args.name}_stage1_frozen",
            lr0=args.lr0,
            optimizer=args.optimizer,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            workers=args.workers,
            patience=args.patience,
            save=args.save,
            verbose=args.verbose,
            freeze=10,  # Freeze backbone layers 0-9
        )
        
        # Unfreeze backbone
        print("\n" + "=" * 60)
        unfreeze_backbone(model)
        print("=" * 60)
        
        # Stage 2: Train with unfrozen backbone
        print("\n" + "=" * 60)
        print(f"Stage 2: Training with unfrozen backbone (epochs {args.freeze+1} to {args.epochs})")
        print("=" * 60)
        
        # Continue training from stage 1 checkpoint
        # Find the last checkpoint from stage 1
        # Note: ultralytics may add numeric suffixes to avoid overwriting (e.g., stage1_frozen, stage1_frozen2, etc.)
        project_path = Path(args.project)
        if not project_path.is_absolute():
            # If relative path, resolve from current working directory
            project_path = Path.cwd() / project_path
        
        stage1_base_name = f"{args.name}_stage1_frozen"
        stage1_dir = None
        resume_from = None
        
        # Try to find the most recent stage1 directory (with or without numeric suffix)
        if project_path.exists():
            # Find all stage1 directories matching the pattern
            stage1_dirs = sorted(
                [d for d in project_path.glob(f"{stage1_base_name}*") if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True  # Most recent first
            )
            
            # Try each stage1 directory until we find one with checkpoints
            for stage1_dir_candidate in stage1_dirs:
                weights_dir = stage1_dir_candidate / "weights"
                if weights_dir.exists():
                    # Priority 1: Look for last.pt (standard ultralytics checkpoint)
                    last_pt = weights_dir / "last.pt"
                    if last_pt.exists():
                        stage1_dir = weights_dir
                        resume_from = str(last_pt.absolute())
                        print(f"Found last checkpoint: {resume_from}")
                        break
                    # Priority 2: Look for best.pt
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        stage1_dir = weights_dir
                        resume_from = str(best_pt.absolute())
                        print(f"Found best checkpoint: {resume_from}")
                        break
                    # Priority 3: Find any .pt file (sorted by modification time)
                    checkpoints = sorted(weights_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
                    if checkpoints:
                        stage1_dir = weights_dir
                        resume_from = str(checkpoints[-1].absolute())
                        print(f"Found checkpoint: {resume_from}")
                        break
        
        if not resume_from:
            # Provide helpful error message
            expected_dir = project_path / stage1_base_name / "weights"
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found.\n"
                f"  Expected location: {expected_dir}\n"
                f"  Searched in: {project_path}\n"
                f"  Please ensure Stage 1 training completed successfully.\n"
                f"  You can check existing stage1 directories with: ls -la {project_path}/*stage1*"
            )
        
        print(f"Resuming Stage 2 training from: {resume_from}")
        print()

        # Re-initialize model with the Stage 1 checkpoint to ensure weights/metadata
        # are loaded correctly. The model created earlier still points to the
        # original pretrain weights, so we create a fresh instance here.
        model_stage2 = YOLO(resume_from)
        # Make sure backbone params are trainable again (stage1 checkpoints carry
        # requires_grad=False flags for frozen layers).
        unfreeze_backbone(model_stage2)
        
        # Ensure data path is absolute and exists
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        data_path = data_path.resolve()
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data configuration file not found: {data_path}")
        
        print(f"Using data configuration: {data_path}")
        print()
        
        remaining_epochs = max(args.epochs - args.freeze, 1)

        results_stage2 = model_stage2.train(
            data=str(data_path),  # Use absolute path
            epochs=remaining_epochs,  # Train remaining epochs with stage1 weights
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=f"{args.name}_stage2_unfrozen",
            lr0=args.lr0,
            optimizer=args.optimizer,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            workers=args.workers,
            patience=args.patience,
            save=args.save,
            verbose=args.verbose,
        )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Stage 1 (frozen): {args.project}/{args.name}_stage1_frozen")
        print(f"Stage 2 (unfrozen): {args.project}/{args.name}_stage2_unfrozen")
        
    else:
        # Mode B (Baseline) or Mode C (Scratch) - single stage training
        # Note: Scratch mode should never freeze layers (parameters are random)
        freeze_param = 0
        
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            lr0=args.lr0,
            optimizer=args.optimizer,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            workers=args.workers,
            patience=args.patience,
            save=args.save,
            verbose=args.verbose,
            freeze=freeze_param,
        )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Results saved to: {args.project}/{args.name}")


if __name__ == '__main__':
    main()

