"""
File: backbone.py
Description:
    Swin Transformer backbone with a projection head for use as a drop-in
    feature extractor in GCN-based multi-label classifiers.
Main Components:
    - SwinBackbone: Wraps a timm Swin Transformer and projects spatial tokens
      to a fixed output dimension compatible with GCN heads.
    - get_swin_backbone: Factory function for common Swin variants.
Inputs:
    images (B, 3, 224, 224)
Outputs:
    image_features (B, out_dim)
Notes:
    - global_pool="" disables timm's built-in pooling so we retain the
      spatial token map for manual AdaptiveAvgPool2d.
    - Designed to replace the ResNet backbone in GCNResnet / ADDGCN.
"""

import timm
import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone with spatial average pooling and linear projection.

    Feature extraction pipeline:
        (B, 3, 224, 224)
        → timm Swin [global_pool=""]  → (B, H, W, C)   # spatial token map
        → permute + AdaptiveAvgPool2d → (B, C)          # global average pool
        → Linear(C, out_dim)          → (B, out_dim)    # project to target dim

    Args:
        model_name (str): timm model identifier for the Swin variant.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
        out_dim (int): Output feature dimension after projection.
            Should match the GCN head's expected input size (default 2048).
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        out_dim: int = 2048,
    ):
        super().__init__()

        # global_pool="" → returns spatial token map (B, H, W, C) instead of scalar
        # num_classes=0  → removes the classification head
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        embed_dim = int(self.swin.num_features)  # 768 for Tiny, 1024 for Base

        # Global spatial average pooling to collapse (H, W) → scalar per channel
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project Swin embedding dim to the shared GCN feature dimension
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract global image features from Swin Transformer with projection.

        Args:
            images (Tensor): Input images.

        Returns:
            Tensor: Projected global image features.

        Shape:
            images:  (B, 3, H, W)
            output:  (B, out_dim)
        """
        # Extract spatial token map from Swin (no built-in pooling)
        tokens = self.swin(images)                          # (B, H, W, C)

        # Permute to channel-first for AdaptiveAvgPool2d compatibility
        tokens = tokens.permute(0, 3, 1, 2)                # (B, C, H, W)

        # Global average pool across spatial dimensions
        image_features = self.pool(tokens).squeeze(-1).squeeze(-1)  # (B, C)

        # Project to target dimension for GCN compatibility
        return self.proj(image_features)                    # (B, out_dim)


def get_swin_backbone(
    model_name: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    out_dim: int = 2048,
) -> SwinBackbone:
    """
    Factory function to instantiate a SwinBackbone.

    Args:
        model_name (str): timm Swin model identifier.
        pretrained (bool): Load ImageNet-pretrained weights if True.
        out_dim (int): Output projection dimension.

    Returns:
        SwinBackbone: Initialized backbone module.
    """
    return SwinBackbone(model_name, pretrained=pretrained, out_dim=out_dim)