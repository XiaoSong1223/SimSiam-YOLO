# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 Backbone Encoder for SimSiam Self-Supervised Learning

This module extracts the YOLOv8 backbone (CSPDarknet) as an encoder,
outputting multi-scale feature maps (P3, P4, P5) for self-supervised learning tasks.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List, Optional, Union

# Add ultralytics to path
_ultralytics_path = Path(__file__).parent.parent / "ultralytics"
if _ultralytics_path.exists():
    sys.path.insert(0, str(_ultralytics_path))

try:
    from ultralytics.nn.tasks import (
        parse_model,
        yaml_model_load,
        torch_safe_load,
        intersect_dicts,
        initialize_weights,
    )
    from ultralytics.utils import LOGGER
except ImportError:
    # Fallback: try importing from installed ultralytics
    from ultralytics.nn.tasks import (
        parse_model,
        yaml_model_load,
        torch_safe_load,
        intersect_dicts,
        initialize_weights,
    )
    from ultralytics.utils import LOGGER


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8 Backbone Encoder for extracting multi-scale features.
    
    This class extracts only the backbone portion of YOLOv8 (CSPDarknet),
    removing the detection head. It outputs three feature maps:
    - P3: stride 8 (1/8 resolution)
    - P4: stride 16 (1/16 resolution)
    - P5: stride 32 (1/32 resolution)
    
    Args:
        cfg (str | dict): Path to YOLOv8 YAML config file or config dictionary.
        weights (str, optional): Path to pretrained .pt weights file.
        ch (int): Number of input channels (default: 3 for RGB).
        verbose (bool): Whether to print model information.
    
    Examples:
        >>> # Load from YAML config
        >>> encoder = YOLOv8Backbone("yolov8n.yaml")
        >>> 
        >>> # Load from YAML with pretrained weights
        >>> encoder = YOLOv8Backbone("yolov8n.yaml", weights="yolov8n.pt")
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(1, 3, 640, 640)
        >>> features = encoder(x)  # Returns [P3, P4, P5]
        >>> print([f.shape for f in features])
    """
    
    def __init__(
        self,
        cfg: Union[str, dict] = "yolov8n.yaml",
        weights: Optional[str] = None,
        ch: int = 3,
        verbose: bool = True,
    ):
        super().__init__()
        
        # Load YAML configuration
        if isinstance(cfg, str):
            # Handle yolov8n.yaml format
            if "yolov8" in cfg.lower() and not cfg.startswith("/"):
                # Try to find in ultralytics cfg directory
                cfg_path = Path(__file__).parent.parent / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8" / cfg
                if not cfg_path.exists():
                    # Try without .yaml extension
                    cfg_path = cfg_path.with_suffix(".yaml")
                if cfg_path.exists():
                    cfg = str(cfg_path)
            self.yaml = yaml_model_load(cfg)
        else:
            self.yaml = cfg
        
        # Extract only backbone configuration
        # Note: parse_model requires both 'backbone' and 'head' keys
        backbone_cfg = {
            "backbone": self.yaml["backbone"],
            "head": [],  # Empty head since we only want backbone
            "nc": self.yaml.get("nc", 80),
            "activation": self.yaml.get("activation"),
            "scales": self.yaml.get("scales"),
            "scale": self.yaml.get("scale", "n"),
        }
        
        # Parse backbone model (only backbone, no head)
        self.model, self.save = parse_model(
            backbone_cfg, ch=ch, verbose=verbose
        )
        
        # Identify P3, P4, P5 layer indices
        # Based on yolov8.yaml structure:
        # - Index 3: P3/8 (stride 8)
        # - Index 5: P4/16 (stride 16)
        # - Index 9: P5/32 (stride 32, after SPPF)
        self.p3_idx = 3  # P3/8
        self.p4_idx = 5  # P4/16
        self.p5_idx = 9  # P5/32
        
        # Initialize weights
        initialize_weights(self)
        
        # Load pretrained weights if provided
        if weights is not None:
            self.load_weights(weights, verbose=verbose)
        
        if verbose:
            self.info()
            LOGGER.info("")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            List[torch.Tensor]: List of three feature maps [P3, P4, P5]:
                - P3: (B, C3, H/8, W/8) - stride 8
                - P4: (B, C4, H/16, W/16) - stride 16
                - P5: (B, C5, H/32, W/32) - stride 32
        """
        y = {}  # Store intermediate outputs (indexed by layer index m.i)
        p3, p4, p5 = None, None, None
        
        for m in self.model:
            # Handle layer connections
            if m.f != -1:  # If not from previous layer
                if isinstance(m.f, int):
                    x = y.get(m.f, x)
                else:
                    # Multiple inputs (for Concat layers, but shouldn't happen in backbone)
                    x = [y.get(j, x) if j != -1 else x for j in m.f]
            
            # Forward through layer
            x = m(x)
            
            # Save output if needed (indexed by m.i)
            if m.i in self.save:
                y[m.i] = x
            
            # Extract P3, P4, P5 features using m.i (layer index)
            if m.i == self.p3_idx:
                p3 = x
            elif m.i == self.p4_idx:
                p4 = x
            elif m.i == self.p5_idx:
                p5 = x
        
        # Return multi-scale features
        return [p3, p4, p5]
    
    def load_weights(self, weights: str, verbose: bool = True):
        """
        Load pretrained weights from a .pt file.
        
        Args:
            weights (str): Path to pretrained weights file.
            verbose (bool): Whether to print loading information.
        """
        ckpt, weights_path = torch_safe_load(weights)
        
        # Extract model from checkpoint
        if isinstance(ckpt, dict):
            model = ckpt.get("model", ckpt.get("ema", None))
            if model is None:
                raise ValueError(f"No model found in checkpoint {weights}")
        else:
            model = ckpt
        
        # Get state dict
        if hasattr(model, "state_dict"):
            csd = model.state_dict()
        elif isinstance(model, dict):
            csd = model
        else:
            raise ValueError(f"Unable to extract state_dict from {weights}")
        
        # Filter to only backbone layers (model.0 to model.9)
        # In YOLOv8, backbone layers are indices 0-9, head layers start from 10+
        backbone_csd = {}
        for k, v in csd.items():
            # Extract layer index from key (format: "model.X.xxx" or "X.xxx")
            key_parts = k.split(".")
            layer_idx = None
            
            # Try to find layer index
            if len(key_parts) > 0:
                # Check if first part is "model" and second is index
                if key_parts[0] == "model" and len(key_parts) > 1:
                    try:
                        layer_idx = int(key_parts[1])
                    except ValueError:
                        pass
                # Or first part is directly the index
                else:
                    try:
                        layer_idx = int(key_parts[0])
                    except ValueError:
                        pass
            
            # Keep only backbone layers (0-9)
            if layer_idx is not None and 0 <= layer_idx <= 9:
                backbone_csd[k] = v
        
        # Intersect with current model state dict
        updated_csd = intersect_dicts(backbone_csd, self.state_dict())
        
        # Load weights
        self.load_state_dict(updated_csd, strict=False)
        
        if verbose:
            LOGGER.info(
                f"Transferred {len(updated_csd)}/{len(self.state_dict())} "
                f"items from pretrained weights {weights_path}"
            )
    
    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
        """
        Print model information.
        
        Args:
            detailed (bool): If True, prints detailed information.
            verbose (bool): If True, prints model information.
            imgsz (int): Image size for FLOPs calculation.
        """
        from ultralytics.utils.torch_utils import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)


if __name__ == "__main__":
    # Example usage
    print("Creating YOLOv8Backbone encoder...")
    
    # Test 1: Load from YAML only
    print("\n1. Loading from YAML config:")
    encoder = YOLOv8Backbone("yolov8n.yaml", verbose=True)
    
    # Test forward pass
    print("\n2. Testing forward pass:")
    x = torch.randn(1, 3, 640, 640)
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"P3 shape (stride 8): {features[0].shape}")
    print(f"P4 shape (stride 16): {features[1].shape}")
    print(f"P5 shape (stride 32): {features[2].shape}")
    
    # Test 2: Load with pretrained weights (if available)
    # Uncomment to test:
    # print("\n3. Loading with pretrained weights:")
    # encoder_pt = YOLOv8Backbone("yolov8n.yaml", weights="yolov8n.pt", verbose=True)
    # features_pt = encoder_pt(x)
    # print("Loaded successfully!")

