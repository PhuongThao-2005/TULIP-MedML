# backbone.py
import timm
import torch
import torch.nn as nn

class SwinBackbone(nn.Module):
    """
    Swin-Tiny backbone với projection head.

    Pipeline:
        (B, 3, 224, 224)
        → timm swin_tiny [global_pool=""]   → (B, 196, 768)   # patch tokens, no pooling
        → AdaptiveAvgPool1d(1)              → (B, 768)         # pool over patches
        → Linear(768, 2048)                 → (B, 2048)        # project lên dim của GCN
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        embed_dim: int = 768,
        out_dim: int = 2048,
    ):
        super().__init__()

        # global_pool="" → trả về feature map (B, C, H, W)
        # num_classes=0  → bỏ classification head
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",   # ← quan trọng: không pool, giữ spatial feature map
        )

        # timm Swin variants have different channel dims:
        # - tiny: num_features = 768
        # - base: num_features = 1024
        in_dim = getattr(self.swin, "num_features", embed_dim)

        # Pool spatial dims → 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))     # input (B, C, H, W) → (B, C, 1, 1)

        # Project 768 → 2048 để align với GCN classifier head
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        tokens = self.swin(x)                        # (B, 7, 7, 768)
        tokens = tokens.permute(0, 3, 1, 2)         # (B, 768, 7, 7)
        pooled = self.pool(tokens)                   # (B, 768, 1, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)      # (B, 768)
        out = self.proj(pooled)                      # (B, 2048)
        return out


def get_swin_backbone(
    model_name: str = "swin_tiny_patch4_window7_224",
    pretrained: bool = True,
    out_dim: int = 2048,
) -> SwinBackbone:
    return SwinBackbone(model_name, pretrained=pretrained, out_dim=out_dim)