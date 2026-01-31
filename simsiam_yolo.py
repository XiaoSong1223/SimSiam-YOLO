# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for Feature-Level SimSiam with YOLOv8 Backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict

from my_experiment.yolo_encoder import YOLOv8Backbone


def build_conv_projector(in_dim: int, out_dim: int = None, hidden_dim: int = None) -> nn.Module:
    """
    Build a 3-layer convolutional projector using 1x1 convolutions.
    
    Args:
        in_dim: Input channel dimension
        out_dim: Output channel dimension (default: same as in_dim)
        hidden_dim: Hidden layer channel dimension (default: same as in_dim)
    
    Returns:
        nn.Sequential: 3-layer projector (Conv2d + BN + ReLU)
    """
    if out_dim is None:
        out_dim = in_dim
    if hidden_dim is None:
        hidden_dim = in_dim
    
    return nn.Sequential(
        # First layer
        nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        # Second layer
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        # Output layer
        nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_dim, affine=False),  # No affine transformation for output BN
    )


def build_conv_predictor(in_dim: int, pred_dim: int = None) -> nn.Module:
    """
    Build a 2-layer convolutional predictor using 1x1 convolutions.
    
    Args:
        in_dim: Input channel dimension
        pred_dim: Hidden/prediction channel dimension (default: in_dim // 4)
    
    Returns:
        nn.Sequential: 2-layer predictor (Conv2d + BN + ReLU)
    """
    if pred_dim is None:
        pred_dim = max(in_dim // 4, 64)  # Default to in_dim // 4, minimum 64
    
    return nn.Sequential(
        # Hidden layer
        nn.Conv2d(in_dim, pred_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(pred_dim),
        nn.ReLU(inplace=True),
        # Output layer
        nn.Conv2d(pred_dim, in_dim, kernel_size=1),  # Output back to in_dim
    )


class SimSiamYOLO(nn.Module):
    """
    Feature-Level SimSiam model with YOLOv8 Backbone.
    
    This model performs dense (feature-level) contrastive learning by:
    1. Using YOLOv8 backbone to extract multi-scale features [P3, P4, P5]
    2. Applying convolutional projectors and predictors to each scale
    3. Maintaining spatial structure throughout (no global pooling)
    
    Args:
        cfg (str | dict): YOLOv8 config file path or dict (default: "yolov8n.yaml")
        weights (str, optional): Path to pretrained YOLOv8 weights
        dim (int): Projector output dimension (default: same as input channels)
        pred_dim (int): Predictor hidden dimension (default: dim // 4)
        shared_heads (bool): Whether to share projector/predictor across scales (default: False)
        verbose (bool): Whether to print model information
    
    Examples:
        >>> # Create model with independent heads for each scale
        >>> model = SimSiamYOLO("yolov8n.yaml", verbose=True)
        >>> 
        >>> # Forward pass with two augmented views
        >>> x1 = torch.randn(2, 3, 640, 640)  # First view
        >>> x2 = torch.randn(2, 3, 640, 640)  # Second view
        >>> outputs = model(x1, x2)
        >>> # outputs is a dict with keys: 'p3', 'p4', 'p5'
        >>> # Each contains: {'p1': tensor, 'p2': tensor, 'z1': tensor, 'z2': tensor}
    """
    
    def __init__(
        self,
        cfg: Union[str, dict] = "yolov8n.yaml",
        weights: Optional[str] = None,
        dim: Optional[int] = None,
        pred_dim: Optional[int] = None,
        shared_heads: bool = False,
        verbose: bool = True,
    ):
        super(SimSiamYOLO, self).__init__()
        
        # Create the encoder (YOLOv8 Backbone)
        self.encoder = YOLOv8Backbone(cfg=cfg, weights=weights, verbose=verbose)
        
        # Get channel dimensions from encoder by running a dummy forward pass
        # This allows us to automatically detect channel dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            dummy_features = self.encoder(dummy_input)
            p3_channels = dummy_features[0].shape[1]
            p4_channels = dummy_features[1].shape[1]
            p5_channels = dummy_features[2].shape[1]
        
        if verbose:
            print(f"Detected channel dimensions: P3={p3_channels}, P4={p4_channels}, P5={p5_channels}")
        
        # Store channel dimensions
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.p5_channels = p5_channels
        
        # Determine output dimensions
        if dim is None:
            # Use input channel dimension as output dimension for each scale
            self.p3_dim = p3_channels
            self.p4_dim = p4_channels
            self.p5_dim = p5_channels
        else:
            # Use specified dimension for all scales
            self.p3_dim = dim
            self.p4_dim = dim
            self.p5_dim = dim
        
        # Determine predictor dimensions
        if pred_dim is None:
            self.p3_pred_dim = max(self.p3_dim // 4, 64)
            self.p4_pred_dim = max(self.p4_dim // 4, 64)
            self.p5_pred_dim = max(self.p5_dim // 4, 64)
        else:
            self.p3_pred_dim = pred_dim
            self.p4_pred_dim = pred_dim
            self.p5_pred_dim = pred_dim
        
        # Build projectors and predictors
        if shared_heads:
            # Shared heads across scales (use P5 dimensions as reference)
            shared_projector = build_conv_projector(p5_channels, self.p5_dim)
            shared_predictor = build_conv_predictor(self.p5_dim, self.p5_pred_dim)
            
            # For P3 and P4, we need to adapt channels first
            self.p3_adapter = nn.Conv2d(p3_channels, p5_channels, kernel_size=1) if p3_channels != p5_channels else nn.Identity()
            self.p4_adapter = nn.Conv2d(p4_channels, p5_channels, kernel_size=1) if p4_channels != p5_channels else nn.Identity()
            
            self.projector_p3 = shared_projector
            self.projector_p4 = shared_projector
            self.projector_p5 = shared_projector
            
            self.predictor_p3 = shared_predictor
            self.predictor_p4 = shared_predictor
            self.predictor_p5 = shared_predictor
        else:
            # Independent heads for each scale (recommended)
            self.projector_p3 = build_conv_projector(p3_channels, self.p3_dim)
            self.projector_p4 = build_conv_projector(p4_channels, self.p4_dim)
            self.projector_p5 = build_conv_projector(p5_channels, self.p5_dim)
            
            self.predictor_p3 = build_conv_predictor(self.p3_dim, self.p3_pred_dim)
            self.predictor_p4 = build_conv_predictor(self.p4_dim, self.p4_pred_dim)
            self.predictor_p5 = build_conv_predictor(self.p5_dim, self.p5_pred_dim)
        
        # Disable bias in the last projector layer (hack: not use bias as it is followed by BN)
        # Note: We already set bias=False in Conv2d, so this is just for reference
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            x1: First view of images, shape (B, C, H, W)
            x2: Second view of images, shape (B, C, H, W)
        
        Returns:
            dict: Dictionary with keys 'p3', 'p4', 'p5', each containing:
                - 'p1': Predictor output for first view, shape (B, C, H', W')
                - 'p2': Predictor output for second view, shape (B, C, H', W')
                - 'z1': Projector output for first view (detached), shape (B, C, H', W')
                - 'z2': Projector output for second view (detached), shape (B, C, H', W')
        """
        # Encode features for both views
        # z1_features: [P3, P4, P5] for x1
        # z2_features: [P3, P4, P5] for x2
        z1_features = self.encoder(x1)
        z2_features = self.encoder(x2)
        
        # Process each scale independently
        outputs = {}
        
        # Process P3 scale
        f1_p3 = z1_features[0]
        f2_p3 = z2_features[0]
        # Apply adapter if using shared heads
        if hasattr(self, 'p3_adapter'):
            f1_p3 = self.p3_adapter(f1_p3)
            f2_p3 = self.p3_adapter(f2_p3)
        z1_p3 = self.projector_p3(f1_p3)
        z2_p3 = self.projector_p3(f2_p3)
        p1_p3 = self.predictor_p3(z1_p3)
        p2_p3 = self.predictor_p3(z2_p3)
        outputs['p3'] = {
            'p1': p1_p3,
            'p2': p2_p3,
            'z1': z1_p3.detach(),
            'z2': z2_p3.detach(),
        }
        
        # Process P4 scale
        f1_p4 = z1_features[1]
        f2_p4 = z2_features[1]
        # Apply adapter if using shared heads
        if hasattr(self, 'p4_adapter'):
            f1_p4 = self.p4_adapter(f1_p4)
            f2_p4 = self.p4_adapter(f2_p4)
        z1_p4 = self.projector_p4(f1_p4)
        z2_p4 = self.projector_p4(f2_p4)
        p1_p4 = self.predictor_p4(z1_p4)
        p2_p4 = self.predictor_p4(z2_p4)
        outputs['p4'] = {
            'p1': p1_p4,
            'p2': p2_p4,
            'z1': z1_p4.detach(),
            'z2': z2_p4.detach(),
        }
        
        # Process P5 scale
        z1_p5 = self.projector_p5(z1_features[2])
        z2_p5 = self.projector_p5(z2_features[2])
        p1_p5 = self.predictor_p5(z1_p5)
        p2_p5 = self.predictor_p5(z2_p5)
        outputs['p5'] = {
            'p1': p1_p5,
            'p2': p2_p5,
            'z1': z1_p5.detach(),
            'z2': z2_p5.detach(),
        }
        
        return outputs


def dense_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute dense (spatial) cosine similarity between two feature maps.
    
    For each spatial location (h, w), compute cosine similarity along the channel dimension,
    then average over all spatial locations.
    
    Args:
        p: Predictor output, shape (N, C, H, W)
        z: Projector output (target), shape (N, C, H, W)
    
    Returns:
        torch.Tensor: Scalar tensor representing average cosine similarity
    """
    # Normalize along channel dimension (dim=1)
    # p_norm: (N, C, H, W) -> normalize along C -> (N, C, H, W)
    # z_norm: (N, C, H, W) -> normalize along C -> (N, C, H, W)
    p_norm = F.normalize(p, p=2, dim=1)
    z_norm = F.normalize(z, p=2, dim=1)
    
    # Compute cosine similarity at each spatial location
    # (p_norm * z_norm): (N, C, H, W)
    # sum along channel dim: (N, H, W)
    cosine_sim = (p_norm * z_norm).sum(dim=1)  # (N, H, W)
    
    # Average over all spatial locations and batch
    return cosine_sim.mean()


def criterion(outputs: Dict[str, Dict[str, torch.Tensor]], reduction: str = 'mean') -> torch.Tensor:
    """
    Compute Dense Cosine Loss for multi-scale Feature-Level SimSiam.
    
    This function computes the negative cosine similarity loss for each scale (P3, P4, P5)
    and sums them together. For each scale, it computes:
        loss = -(cosine_sim(p1, z2).mean() + cosine_sim(p2, z1).mean()) * 0.5
    
    Args:
        outputs: Dictionary with keys 'p3', 'p4', 'p5', each containing:
            - 'p1': Predictor output for first view, shape (N, C, H, W)
            - 'p2': Predictor output for second view, shape (N, C, H, W)
            - 'z1': Projector output for first view (detached), shape (N, C, H, W)
            - 'z2': Projector output for second view (detached), shape (N, C, H, W)
        reduction: 'mean' or 'sum' (default: 'mean')
    
    Returns:
        torch.Tensor: Total loss across all scales
    
    Examples:
        >>> model = SimSiamYOLO("yolov8n.yaml")
        >>> x1 = torch.randn(2, 3, 640, 640)
        >>> x2 = torch.randn(2, 3, 640, 640)
        >>> outputs = model(x1, x2)
        >>> loss = criterion(outputs)
        >>> loss.backward()
    """
    total_loss = 0.0
    scale_losses = {}
    
    # Process each scale
    for scale in ['p3', 'p4', 'p5']:
        if scale not in outputs:
            continue
        
        p1 = outputs[scale]['p1']
        p2 = outputs[scale]['p2']
        z1 = outputs[scale]['z1']
        z2 = outputs[scale]['z2']
        
        # Compute cosine similarity for both directions
        # p1 should match z2, p2 should match z1
        sim_12 = dense_cosine_similarity(p1, z2)  # p1 -> z2
        sim_21 = dense_cosine_similarity(p2, z1)  # p2 -> z1
        
        # Negative cosine similarity loss (we want to maximize similarity, so minimize negative)
        scale_loss = -(sim_12 + sim_21) * 0.5
        scale_losses[scale] = scale_loss
        total_loss = total_loss + scale_loss
    
    if reduction == 'mean':
        # Average over scales
        num_scales = len(scale_losses)
        if num_scales > 0:
            total_loss = total_loss / num_scales
    # else: 'sum' - already summed
    
    return total_loss


if __name__ == "__main__":
    # Example usage
    print("Creating SimSiamYOLO model...")
    
    # Create model
    model = SimSiamYOLO("yolov8n.yaml", verbose=True)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x1 = torch.randn(2, 3, 640, 640)
    x2 = torch.randn(2, 3, 640, 640)
    
    outputs = model(x1, x2)
    
    print("\nOutput shapes:")
    for scale in ['p3', 'p4', 'p5']:
        print(f"\n{scale.upper()}:")
        for key in ['p1', 'p2', 'z1', 'z2']:
            shape = outputs[scale][key].shape
            print(f"  {key}: {shape}")
    
    # Test loss computation
    print("\nTesting Dense Cosine Loss...")
    loss = criterion(outputs)
    print(f"Total loss: {loss.item():.4f}")
    
    # Test individual scale losses
    print("\nIndividual scale losses:")
    for scale in ['p3', 'p4', 'p5']:
        scale_outputs = {scale: outputs[scale]}
        scale_loss = criterion(scale_outputs)
        print(f"  {scale.upper()}: {scale_loss.item():.4f}")
    
    print("\nModel and loss function created successfully!")

