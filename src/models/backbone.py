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

        # global_pool="" → trả về full patch tokens (B, num_patches, C)
        # num_classes=0  → bỏ classification head
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",   # ← quan trọng: không pool, giữ spatial tokens
        )

        # Pool 196 patch tokens → 1 vector
        self.pool = nn.AdaptiveAvgPool1d(1)          # input (B, C, L) → (B, C, 1)

        # Project 768 → 2048 để align với GCN classifier head
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        tokens = self.swin(x)                        # (B, 196, 768)
        pooled = self.pool(tokens.transpose(1, 2))   # (B, 768, 196) → (B, 768, 1)
        pooled = pooled.squeeze(-1)                  # (B, 768)
        out = self.proj(pooled)                      # (B, 2048)
        return out


def get_swin_backbone(
    model_name: str = "swin_tiny_patch4_window7_224",
    pretrained: bool = True,
    out_dim: int = 2048,
) -> SwinBackbone:
    return SwinBackbone(model_name, pretrained=pretrained, out_dim=out_dim)


# ── Quick shape verification ──────────────────────────────────────────────────
if __name__ == "__main__":
    backbone = get_swin_backbone(pretrained=False)  # pretrained=False để test nhanh
    backbone.eval()

    dummy = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        tokens = backbone.swin(dummy)
        print(f"After swin (patch tokens) : {tuple(tokens.shape)}")   # (2, 196, 768)

        pooled = backbone.pool(tokens.transpose(1, 2)).squeeze(-1)
        print(f"After AdaptiveAvgPool1d   : {tuple(pooled.shape)}")   # (2, 768)

        out = backbone.proj(pooled)
        print(f"Final output              : {tuple(out.shape)}")       # (2, 2048)

    print("\n✓ Shape confirmed: (B,3,224,224) → (B,2048)")