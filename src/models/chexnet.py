# src/models/chexnet.py
"""
CheXNet baseline cho CheXpert fine-tuning.

- Backbone: torchvision DenseNet-121, pretrained ImageNet (không load CheXNet .pth)
- Classifier: Linear(1024 → num_classes) cho 14 nhãn CheXpert
- Output: logits (BCEWithLogitsLoss trong train_chexnet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from src.data.chexpert import CHEXPERT_CLASSES

NUM_CLASSES = len(CHEXPERT_CLASSES)


def _densenet121_backbone(pretrained: bool):
    """DenseNet-121 với weights ImageNet nếu pretrained=True."""
    if not pretrained:
        return torchvision.models.densenet121(weights=None)
    try:
        from torchvision.models import DenseNet121_Weights

        w = DenseNet121_Weights.IMAGENET1K_V1
        return torchvision.models.densenet121(weights=w)
    except (ImportError, AttributeError):
        return torchvision.models.densenet121(pretrained=True)


class CheXNetBaseline(nn.Module):
    """
    DenseNet-121 + linear head đa nhãn.

    Input : (B, 3, H, W)
    Output: (B, num_classes) logits
    """

    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        densenet = _densenet121_backbone(pretrained=pretrained)
        num_ftrs = densenet.classifier.in_features

        self.features = densenet.features
        self.classifier = nn.Linear(num_ftrs, num_classes)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                   # (B, 1024, H', W')
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

    def parameters_to_optimize(self):
        return self.parameters()


def build_chexnet(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    ckpt_path: str | None = None,
) -> CheXNetBaseline:
    """
    Khởi tạo DenseNet-121 baseline.

    Args:
        num_classes: số lớp (14 CheXpert)
        pretrained: True → ImageNet weights từ torchvision
        ckpt_path: bỏ qua (giữ tham số để tương thích config cũ; không load file)
    """
    if ckpt_path:
        print(
            f"[CheXNet] ckpt_path={ckpt_path!r} được bỏ qua — "
            "chỉ dùng DenseNet-121 pretrained ImageNet từ torchvision."
        )
    print(f"[CheXNet] DenseNet-121, pretrained(ImageNet)={pretrained}")
    return CheXNetBaseline(num_classes=num_classes, pretrained=pretrained)
