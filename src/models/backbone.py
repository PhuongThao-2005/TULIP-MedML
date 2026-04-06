# backbone.py
import timm
import torch
import torch.nn as nn

class SwinBackbone(nn.Module):
    """
    Swin backbone (timm) + projection to GCN feature dim.

    Pipeline:
        (B, 3, 224, 224)
        → timm Swin [global_pool=""]  → (B, 7, 7, C)   # C = num_features (768 Tiny, 1024 Base)
        → permute + AdaptiveAvgPool2d  → (B, C)
        → Linear(C, out_dim)           → (B, out_dim)
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        out_dim: int = 2048,
    ):
        super().__init__()

        # global_pool="" → spatial map (B, H, W, C) from timm Swin
        # num_classes=0  → bỏ classification head
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        embed_dim = int(self.swin.num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.swin(x)                        # (B, H, W, C)
        tokens = tokens.permute(0, 3, 1, 2)         # (B, C, H, W)
        pooled = self.pool(tokens).squeeze(-1).squeeze(-1)  # (B, C)
        return self.proj(pooled)                     # (B, out_dim)


def get_swin_backbone(
    model_name: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    out_dim: int = 2048,
) -> SwinBackbone:
    return SwinBackbone(model_name, pretrained=pretrained, out_dim=out_dim)