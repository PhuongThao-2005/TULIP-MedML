import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) — công thức gốc từ Ben-Baruch et al. 2021.
    https://arxiv.org/abs/2009.14119

    Args:
        gamma_pos (float): focusing parameter cho positive samples (thường = 0).
        gamma_neg (float): focusing parameter cho negative samples (thường = 4).
        margin    (float): probability margin shift cho negative branch (thường = 0.05).
        reduction (str)  : 'mean' | 'sum' | 'none'.
        disable_torch_grad_focal_loss (bool): tắt grad qua focal weight để ổn định.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
        reduction: str = 'mean',
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = 1e-8

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f'logits and targets phải cùng shape, '
                f'got {tuple(logits.shape)} vs {tuple(targets.shape)}'
            )

        # Remap uncertain (-1) → negative (0)
        targets = targets.clone().float()
        targets[targets < 0] = 0.0

        p = torch.sigmoid(logits)

        # ── Positive branch: -(1-p)^γ+ · log(p) ────────────────────────────
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_pos = torch.pow(1.0 - p, self.gamma_pos)
        else:
            focal_pos = torch.pow(1.0 - p, self.gamma_pos)

        loss_pos = -focal_pos * torch.log(p.clamp(min=self.eps))

        # ── Negative branch với margin shift: -(p_m)^γ- · log(1-p_m) ───────
        p_m = torch.clamp(p - self.margin, min=0.0)
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_neg = torch.pow(p_m, self.gamma_neg)
        else:
            focal_neg = torch.pow(p_m, self.gamma_neg)

        loss_neg = -focal_neg * torch.log((1.0 - p_m).clamp(min=self.eps))

        # ── Aggregate ────────────────────────────────────────────────────────
        loss = targets * loss_pos + (1.0 - targets) * loss_neg   # element-wise
        total_loss = loss.sum()

        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return total_loss
        if self.reduction == 'mean':
            return total_loss / logits.size(0)
        raise ValueError(f'Unsupported reduction: {self.reduction!r}')