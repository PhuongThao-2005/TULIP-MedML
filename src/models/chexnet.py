# src/models/chexnet_baseline.py
"""
CheXNet baseline cho CheXpert fine-tuning.

Architecture: DenseNet-121 (giống CheXNet gốc)
- Backbone: load từ model.pth.tar (pretrained ChestX-ray14)
- Classifier: replace Linear(1024 → 14) cho CheXpert labels
- Output: raw LOGITS (không có Sigmoid) — khớp với evaluate.py
  của repo (evaluate.py tự gọi torch.sigmoid bên trong)

Tại sao LOGITS thay vì Sigmoid như CheXNet gốc?
  CheXNet gốc dùng Sigmoid+BCELoss.
  Repo này dùng BCEWithLogitsLoss (numerically stable hơn)
  → bỏ Sigmoid ở model, loss tự lo.
"""

import torch
import torch.nn as nn
import torchvision
from src.data.chexpert import CHEXPERT_CLASSES

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — khớp với chexpert.py
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = len(CHEXPERT_CLASSES)   # 14
# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class CheXNetBaseline(nn.Module):
    """
    DenseNet-121 fine-tuned cho CheXpert.

    Input : (B, 3, H, W)  — normalized RGB
    Output: (B, 14)        — raw logits (NO sigmoid)

    evaluate.py của repo gọi torch.sigmoid(logits) bên trong
    → không cần sigmoid ở đây, giữ nhất quán với C1/C5.
    """

    # Dùng ImageNet normalization — giống C1/C5 trong engine.py
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std  = [0.229, 0.224, 0.225]

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        densenet = torchvision.models.densenet121(weights=None)
        num_ftrs = densenet.classifier.in_features   # 1024

        # Giữ toàn bộ features, replace classifier
        self.features   = densenet.features
        self.classifier = nn.Linear(num_ftrs, num_classes)

        # Khởi tạo classifier head theo Xavier (chuẩn cho fine-tuning)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                   # (B, 1024, H', W')
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)        # (B, 1024)
        return self.classifier(feat)              # (B, 14) — raw logits

    # def get_config_optim(self, lr: float, lrp: float):
    #     """
    #     Trả về param groups với learning rate khác nhau:
    #       - backbone (features): lr * lrp  — fine-tune chậm hơn
    #       - classifier head    : lr        — học nhanh hơn

    #     Khớp với build_optimizer() trong train.py của repo
    #     (SGD với momentum + weight_decay).

    #     lrp thường = 0.1 → backbone lr = lr/10
    #     """
    #     return [
    #         {'params': self.features.parameters(),   'lr': lr * lrp},
    #         {'params': self.classifier.parameters(), 'lr': lr},
    #     ]

    def parameters_to_optimize(self):
        """
        CheXNet gốc: Adam với lr đồng nhất toàn model.
        Trả về tất cả parameters (không chia backbone/head).
        """
        return self.parameters()


# ─────────────────────────────────────────────────────────────────────────────
#  Backbone loader — Option B
# ─────────────────────────────────────────────────────────────────────────────

def build_chexnet(ckpt_path: str,
                  num_classes: int = NUM_CLASSES) -> CheXNetBaseline:
    """
    Tạo CheXNetBaseline, load backbone từ model.pth.tar.

    Quy trình:
      1. Tạo model mới với classifier cho CheXpert (14 class)
      2. Load state_dict từ checkpoint ChestX-ray14
      3. strict=False → bỏ qua classifier mismatch (14 class cũ ≠ 14 class mới,
         tên nhãn khác nhau hoàn toàn)
      4. Classifier head giữ nguyên Xavier init từ bước __init__

    Args:
        ckpt_path  : đường dẫn tới model.pth.tar
        num_classes: 14 (CheXpert)

    Returns:
        CheXNetBaseline với backbone đã load
    """
    model = CheXNetBaseline(num_classes=num_classes)

    print(f"[CheXNet] Loading backbone from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Checkpoint gốc được wrap bằng DataParallel → bỏ prefix "module."
    raw_state = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

    # Checkpoint gốc: densenet121.features.* và densenet121.classifier.*
    # Model mới:      features.* và classifier.*
    # → remap key
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("densenet121.features."):
            new_k = k.replace("densenet121.features.", "features.")
            remapped[new_k] = v
        elif k.startswith("densenet121.classifier."):
            # Bỏ qua — classifier cũ (14 ChestX-ray14 classes) không dùng
            pass
        else:
            remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Chỉ classifier keys được phép missing
    bad_missing = [k for k in missing if "classifier" not in k]
    if bad_missing:
        print(f"[CheXNet] WARNING — non-classifier missing keys:\n  {bad_missing}")
    else:
        n_classifier = len([k for k in missing if "classifier" in k])
        print(f"[CheXNet] Backbone loaded OK. "
              f"Classifier re-initialized ({n_classifier} keys). "
              f"Unexpected keys ignored: {len(unexpected)}")

    return model