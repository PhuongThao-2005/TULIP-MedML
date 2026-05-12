"""
File: chexnet.py
Description:
    CheXNet baseline for CheXpert multi-label classification.
    Uses a DenseNet-121 backbone pretrained on ImageNet (not the original
    Stanford CheXNet weights) with a linear classification head.
Main Components:
    - CheXNetBaseline: DenseNet-121 + linear head for 14-class multi-label output.
    - build_chexnet: Factory function for constructing the model.
Inputs:
    images (B, 3, H, W)
Outputs:
    logits (B, num_classes) — raw scores, apply BCEWithLogitsLoss during training.
Notes:
    - Output is logits, not probabilities. Apply sigmoid at inference.
    - ckpt_path argument is accepted for config compatibility but intentionally
      ignored — weights come from torchvision (ImageNet), not an external file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from src.data.chexpert import CHEXPERT_CLASSES

NUM_CLASSES = len(CHEXPERT_CLASSES)


# Backbone

def _build_densenet121(pretrained: bool) -> torchvision.models.DenseNet:
    """
    Instantiate DenseNet-121 with optional ImageNet weights.

    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 weights from torchvision.

    Returns:
        DenseNet: DenseNet-121 model with original FC head (replaced downstream).
    """
    if not pretrained:
        return torchvision.models.densenet121(weights=None)

    try:
        from torchvision.models import DenseNet121_Weights
        return torchvision.models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )
    except (ImportError, AttributeError):
        # Fallback for older torchvision versions
        return torchvision.models.densenet121(pretrained=True)

# Model

class CheXNetBaseline(nn.Module):
    """
    DenseNet-121 multi-label classifier for CheXpert.

    The original DenseNet FC head is replaced with a linear layer projecting
    from 1024 features to num_classes logits.

    Args:
        num_classes (int): Number of output classes (default: 14 CheXpert labels).
        pretrained (bool): Load ImageNet-pretrained DenseNet-121 weights.

    Notes:
        - Normalization constants match ImageNet statistics used during pretraining.
        - Xavier uniform initialization is applied to the classifier head.
    """

    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()
        densenet = _build_densenet121(pretrained=pretrained)
        num_features = densenet.classifier.in_features  # 1024

        # Keep only the feature extraction layers; discard the classification head
        self.features = densenet.features

        # Multi-label linear head: outputs raw logits for BCEWithLogitsLoss
        self.classifier = nn.Linear(num_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features and project to class logits.

        Args:
            images (Tensor): Preprocessed input images.

        Returns:
            Tensor: Raw class logits (pre-sigmoid).

        Shape:
            images: (B, 3, H, W)
            output: (B, num_classes)
        """
        # Extract dense feature maps from DenseNet feature extractor
        image_features = self.features(images)              # (B, 1024, H', W')

        # Global average pool to get a fixed-size representation
        image_features = nn.functional.adaptive_avg_pool2d(
            image_features, (1, 1)
        )                                                   # (B, 1024, 1, 1)
        image_features = image_features.view(
            image_features.size(0), -1
        )                                                   # (B, 1024)

        # Project to per-class logits
        logits = self.classifier(image_features)            # (B, num_classes)
        return logits

    def parameters_to_optimize(self):
        """Return all model parameters (no per-group LR split for this baseline)."""
        return self.parameters()


# Factory

def build_chexnet(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    ckpt_path: str | None = None,
) -> CheXNetBaseline:
    """
    Construct and return a CheXNetBaseline model.

    Args:
        num_classes (int): Number of output classes (14 for CheXpert).
        pretrained (bool): If True, initialize with ImageNet weights from torchvision.
        ckpt_path (str, optional): Accepted for config compatibility but ignored.
            External checkpoint loading is not performed here to keep the
            baseline reproducible from standard torchvision weights.

    Returns:
        CheXNetBaseline: Initialized DenseNet-121 multi-label classifier.
    """
    if ckpt_path:
        print(
            f"[CheXNet] ckpt_path={ckpt_path!r} is ignored — "
            "using DenseNet-121 pretrained on ImageNet via torchvision."
        )

    print(f"[CheXNet] DenseNet-121, pretrained(ImageNet)={pretrained}")
    return CheXNetBaseline(num_classes=num_classes, pretrained=pretrained)