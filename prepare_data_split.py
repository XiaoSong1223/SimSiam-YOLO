#!/usr/bin/env python
"""
Prepare subset of data from TT-100K dataset for limited labeled data scenario.

This script:
1. Randomly samples 10%, 30%, 50% of training images and labels
2. Creates a new directory structure for the 10%, 30%, 50% dataset
3. Copies selected images and labels to the new directories
4. Generates a new YAML configuration file
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import yaml


def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_config(config, yaml_path):
    """Save YAML configuration file."""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_image_label_pairs(train_images_dir, train_labels_dir):
    """
    Get all image-label pairs from training directory.
    
    Returns:
        List of tuples: [(image_path, label_path), ...]
    """
    image_dir = Path(train_images_dir)
    label_dir = Path(train_labels_dir)
    
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_file in image_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            # Find corresponding label file
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"Warning: No label file found for {img_file.name}")
    
    return pairs


def sample_data_pairs(pairs, percentage=0.1, seed=None):
    """
    Randomly sample a percentage of data pairs.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        percentage: Percentage to sample (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled pairs
    """
    if seed is not None:
        random.seed(seed)
    
    num_samples = max(1, int(len(pairs) * percentage))
    sampled = random.sample(pairs, num_samples)
    
    return sampled


def create_directory_structure(output_base_dir):
    """Create directory structure for 10% dataset."""
    base = Path(output_base_dir)
    
    dirs = [
        base / "images" / "train",
        base / "labels" / "train",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return base


def copy_files(pairs, output_images_dir, output_labels_dir, use_symlink=False):
    """
    Copy or symlink image and label files to output directories.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        output_images_dir: Output directory for images
        output_labels_dir: Output directory for labels
        use_symlink: If True, create symlinks instead of copying
    """
    output_images = Path(output_images_dir)
    output_labels = Path(output_labels_dir)
    
    copied_images = 0
    copied_labels = 0
    
    for img_path, label_path in pairs:
        # Copy/symlink image
        dest_img = output_images / img_path.name
        if use_symlink:
            if dest_img.exists():
                dest_img.unlink()
            dest_img.symlink_to(img_path.absolute())
        else:
            shutil.copy2(img_path, dest_img)
        copied_images += 1
        
        # Copy/symlink label
        dest_label = output_labels / label_path.name
        if use_symlink:
            if dest_label.exists():
                dest_label.unlink()
            dest_label.symlink_to(label_path.absolute())
        else:
            shutil.copy2(label_path, dest_label)
        copied_labels += 1
        
        if (copied_images + copied_labels) % 100 == 0:
            print(f"  Processed {copied_images} images and {copied_labels} labels...")
    
    print(f"Copied {copied_images} images and {copied_labels} labels")
    return copied_images, copied_labels


def create_yaml_config(original_yaml_path, output_yaml_path, output_base_dir, 
                       train_images_dir, val_images_dir=None):
    """
    Create new YAML configuration file for 10% dataset.
    
    Args:
        original_yaml_path: Path to original data.yaml
        output_yaml_path: Path to save new YAML file
        output_base_dir: Base directory of the new dataset
        train_images_dir: Path to training images directory
        val_images_dir: Path to validation images directory (optional)
    """
    # Load original config
    original_config = load_yaml_config(original_yaml_path)
    
    # Create new config
    new_config = {
        'path': str(Path(output_base_dir).absolute()),
        'train': 'images/train',
        'val': val_images_dir if val_images_dir else original_config.get('val', 'test/images'),
    }
    
    # Copy class information
    if 'nc' in original_config:
        new_config['nc'] = original_config['nc']
    if 'names' in original_config:
        new_config['names'] = original_config['names']
    
    # Save new config
    save_yaml_config(new_config, output_yaml_path)
    print(f"Created YAML config: {output_yaml_path}")
    
    return new_config


def main():
    parser = argparse.ArgumentParser(
        description='Prepare 10% data split from TT-100K dataset'
    )
    parser.add_argument(
        '--train-images',
        type=str,
        default='../dataset/tt100k_2021/train/images',
        help='Path to training images directory'
    )
    parser.add_argument(
        '--train-labels',
        type=str,
        default='../dataset/tt100k_2021/train/labels',
        help='Path to training labels directory'
    )
    parser.add_argument(
        '--val-images',
        type=str,
        default=None,
        help='Path to validation images directory (default: use test/images)'
    )
    parser.add_argument(
        '--original-yaml',
        type=str,
        default='../dataset/tt100k_2021/data.yaml',
        help='Path to original data.yaml file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../dataset/tt100k_10pct',
        help='Output directory for 10% dataset'
    )
    parser.add_argument(
        '--percentage',
        type=float,
        default=0.1,
        help='Percentage of data to sample (default: 0.1 for 10%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Use symlinks instead of copying files (saves disk space)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TT-100K 10% Data Split Preparation")
    print("=" * 60)
    print()
    
    # Step 1: Get all image-label pairs
    print(f"[Step 1] Scanning training data...")
    print(f"  Images directory: {args.train_images}")
    print(f"  Labels directory: {args.train_labels}")
    
    pairs = get_image_label_pairs(args.train_images, args.train_labels)
    print(f"  Found {len(pairs)} image-label pairs")
    print()
    
    # Step 2: Sample 10% of data
    print(f"[Step 2] Randomly sampling {args.percentage*100:.1f}% of data...")
    print(f"  Random seed: {args.seed}")
    sampled_pairs = sample_data_pairs(pairs, args.percentage, args.seed)
    print(f"  Sampled {len(sampled_pairs)} pairs ({len(sampled_pairs)/len(pairs)*100:.2f}%)")
    print()
    
    # Step 3: Create directory structure
    print(f"[Step 3] Creating directory structure...")
    output_base = create_directory_structure(args.output_dir)
    print()
    
    # Step 4: Copy/symlink files
    print(f"[Step 4] {'Creating symlinks' if args.symlink else 'Copying files'}...")
    output_images_dir = output_base / "images" / "train"
    output_labels_dir = output_base / "labels" / "train"
    
    copied_images, copied_labels = copy_files(
        sampled_pairs,
        output_images_dir,
        output_labels_dir,
        use_symlink=args.symlink
    )
    print()
    
    # Step 5: Create YAML config
    print(f"[Step 5] Creating YAML configuration...")
    output_yaml = output_base / "data_10pct.yaml"
    
    val_images = args.val_images
    if val_images is None:
        # Use test images as validation
        test_images_dir = Path(args.train_images).parent.parent / "test" / "images"
        if test_images_dir.exists():
            val_images = str(test_images_dir.absolute())
            print(f"  Using test images as validation: {val_images}")
        else:
            print(f"  Warning: Test images directory not found, using original val path")
    
    new_config = create_yaml_config(
        args.original_yaml,
        str(output_yaml),
        str(output_base.absolute()),
        str(output_images_dir.absolute()),
        val_images
    )
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original training pairs: {len(pairs)}")
    print(f"Sampled pairs: {len(sampled_pairs)} ({args.percentage*100:.1f}%)")
    print(f"Copied images: {copied_images}")
    print(f"Copied labels: {copied_labels}")
    print(f"Output directory: {output_base.absolute()}")
    print(f"YAML config: {output_yaml.absolute()}")
    print()
    print("Dataset structure:")
    print(f"  Train images: {output_images_dir.absolute()}")
    print(f"  Train labels: {output_labels_dir.absolute()}")
    print(f"  Val images: {new_config.get('val', 'N/A')}")
    print()
    print("=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

