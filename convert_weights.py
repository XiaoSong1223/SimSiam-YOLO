#!/usr/bin/env python
"""
Convert SimSiamYOLO checkpoint to YOLOv8 compatible format.

This script extracts the backbone weights from a trained SimSiamYOLO model
and converts them to a format that can be loaded by YOLOv8 for fine-tuning.
"""

import argparse
import torch
from pathlib import Path
from copy import deepcopy
from datetime import datetime

# Import ultralytics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ultralytics"))

try:
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version
except ImportError:
    print("Warning: ultralytics not found. Please install it or add to path.")
    ultralytics_version = "unknown"


def extract_backbone_weights(checkpoint_path: str, verbose: bool = True) -> dict:
    """
    Extract backbone weights from SimSiamYOLO checkpoint.
    
    Args:
        checkpoint_path: Path to SimSiamYOLO checkpoint file
        verbose: Whether to print extraction information
    
    Returns:
        Dictionary containing backbone weights with YOLOv8-compatible keys
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint
    
    if verbose:
        print(f"Total keys in checkpoint: {len(state_dict)}")
    
    # Extract encoder (backbone) weights
    # In SimSiamYOLO, encoder weights have prefix "encoder.model."
    # We need to convert "encoder.model.X" -> "model.X" for YOLOv8 compatibility
    backbone_weights = {}
    encoder_prefix = "encoder.model."
    
    for key, value in state_dict.items():
        if key.startswith(encoder_prefix):
            # Remove "encoder." prefix to get "model.X"
            new_key = key[len("encoder."):]  # Remove "encoder." prefix
            backbone_weights[new_key] = value
            if verbose and len(backbone_weights) <= 10:  # Print first 10 keys
                print(f"  {key} -> {new_key}")
    
    if verbose:
        print(f"Extracted {len(backbone_weights)} backbone weights")
        # Extract layer indices
        layer_indices = sorted(set([
            int(k.split('.')[1]) for k in backbone_weights.keys() 
            if k.startswith('model.') and k.split('.')[1].isdigit()
        ]))
        print(f"Backbone layer indices: {layer_indices}")
    
    return backbone_weights


def create_yolo_checkpoint(backbone_weights: dict, yolo_model, checkpoint_path: str = None) -> dict:
    """
    Create a YOLOv8-compatible checkpoint with backbone weights.
    
    Args:
        backbone_weights: Dictionary of backbone weights
        yolo_model: YOLOv8 model instance
        checkpoint_path: Optional path to original checkpoint (for metadata)
    
    Returns:
        Dictionary in YOLOv8 checkpoint format
    """
    # Get YOLOv8 model's state dict
    yolo_state_dict = yolo_model.model.state_dict()
    
    # Match and update backbone weights
    matched_keys = []
    unmatched_keys = []
    
    for key, value in backbone_weights.items():
        if key in yolo_state_dict:
            # Check shape compatibility
            if yolo_state_dict[key].shape == value.shape:
                yolo_state_dict[key] = value
                matched_keys.append(key)
            else:
                print(f"Warning: Shape mismatch for {key}: "
                      f"YOLOv8 {yolo_state_dict[key].shape} vs Backbone {value.shape}")
                unmatched_keys.append(key)
        else:
            unmatched_keys.append(key)
    
    print(f"\nWeight matching summary:")
    print(f"  Matched: {len(matched_keys)} keys")
    print(f"  Unmatched: {len(unmatched_keys)} keys")
    
    if unmatched_keys and len(unmatched_keys) <= 20:
        print(f"  Unmatched keys: {unmatched_keys[:20]}")
    
    # Update model with matched weights
    yolo_model.model.load_state_dict(yolo_state_dict, strict=False)
    
    # Create checkpoint dictionary
    ckpt = {
        'model': deepcopy(yolo_model.model).half(),  # Save as FP16
        'date': datetime.now().isoformat(),
        'version': ultralytics_version,
        'license': 'AGPL-3.0',
        'docs': 'https://docs.ultralytics.com',
    }
    
    # Add metadata from original checkpoint if available
    if checkpoint_path:
        original_ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'args' in original_ckpt:
            ckpt['train_args'] = original_ckpt['args']
        if 'epoch' in original_ckpt:
            ckpt['epoch'] = original_ckpt['epoch']
            ckpt['best_fitness'] = None  # No fitness for self-supervised model
    
    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert SimSiamYOLO checkpoint to YOLOv8 format'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to SimSiamYOLO checkpoint file (checkpoint.pth.tar)'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='yolov8n.yaml',
        help='YOLOv8 config file (default: yolov8n.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yolov8_simsiam_pretrained.pt',
        help='Output filename (default: yolov8_simsiam_pretrained.pt)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    # Check input file
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print("=" * 60)
    print("SimSiamYOLO to YOLOv8 Weight Converter")
    print("=" * 60)
    
    # Step 1: Extract backbone weights
    print("\n[Step 1] Extracting backbone weights...")
    backbone_weights = extract_backbone_weights(str(checkpoint_path), verbose=args.verbose)
    
    if len(backbone_weights) == 0:
        raise ValueError("No backbone weights found! Check checkpoint structure.")
    
    # Step 2: Create YOLOv8 model
    print(f"\n[Step 2] Creating YOLOv8 model from {args.cfg}...")
    yolo_model = YOLO(args.cfg)
    print(f"YOLOv8 model created: {yolo_model.model_name}")
    
    # Step 3: Create checkpoint with backbone weights
    print("\n[Step 3] Creating YOLOv8-compatible checkpoint...")
    ckpt = create_yolo_checkpoint(backbone_weights, yolo_model, str(checkpoint_path))
    
    # Step 4: Save checkpoint
    print(f"\n[Step 4] Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print(f"\nYou can now use the converted model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{args.output}')")
    print(f"  model.train(data='your_dataset.yaml', epochs=100)")
    print()


if __name__ == '__main__':
    main()

