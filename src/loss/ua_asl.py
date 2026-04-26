"""
File: ua_asl.py
Description:
    Uncertainty-Aware Asymmetric Sigmoid Loss (UA-ASL) for multi-label
    classification with uncertain ground-truth annotations.
Main Components:
    - UncertaintyAwareASL: Loss module handling targets in {-1, 0, +1}.
Inputs:
    logits (B, C) and targets (B, C) with values in {-1, 0, 1}.
Outputs:
    Scalar loss (or per-element tensor when reduction="none").
Notes:
    - Extends Asymmetric Loss (ASL) with a soft interpolation term for -1 labels.
    - The uncertain term does NOT apply the margin shift used for negatives,
      because the shift would zero out gradients for low-probability uncertain
      positives, eliminating their training signal entirely.
    - Related to: src/models/*.py (consumed by train scripts via criterion).
"""

import torch
import torch.nn as nn


class UncertaintyAwareASL(nn.Module):
    """
    Uncertainty-Aware Asymmetric Sigmoid Loss for multi-label classification.

    Targets are expected in {-1, 0, 1}:
        +1  : confirmed positive
         0  : confirmed negative
        -1  : uncertain (softly interpolated between positive and negative loss)

    Args:
        gamma_pos (float): Focal exponent applied to positive samples.
            y+ = 0 reduces the positive term to standard cross-entropy.
        gamma_neg (float): Focal exponent applied to negative samples.
            Higher values suppress easy negatives more aggressively.
        margin (float): Probability margin shift for negative samples.
            Shifts p down by `margin` before computing negative loss,
            effectively ignoring very low-confidence negative predictions.
        lambda_unc (float): Weight scaling the uncertain loss term relative
            to the confirmed positive and negative terms.
        alpha (float): Interpolation weight for uncertain samples.
            Uncertain loss = alpha * pos_term + (1 - alpha) * neg_term.
        eps (float): Small constant for numerical stability in log computations.
        reduction (str): "mean", "sum", or "none".
        disable_torch_grad_focal_loss (bool): Detach focal weights from the
            computation graph to avoid second-order gradient overhead.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
        lambda_unc: float = 0.5,
        alpha: float = 0.5,
        eps: float = 1e-8,
        reduction: str = "mean",
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.lambda_unc = lambda_unc
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the uncertainty-aware asymmetric loss.

        Args:
            logits (Tensor): Raw model outputs (before sigmoid).
            targets (Tensor): Ground-truth labels in {-1, 0, 1}.

        Returns:
            Tensor: Scalar loss (or per-element tensor if reduction="none").

        Shape:
            logits:  (B, C)
            targets: (B, C)
            output:  scalar, or (B, C) when reduction="none"

        Notes:
            - Uncertain samples (-1) are NOT treated as positives or negatives.
              Instead they receive a soft blend of both loss terms.
            - The margin shift is intentionally excluded from the uncertain
              term to preserve gradient flow for uncertain low-p predictions.
              (Giải thích: margin shift bằng 0 nên uncertain luôn có gradient)
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, "
                f"got {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )

        targets = targets.float()
        probs = torch.sigmoid(logits)  # (B, C)

        # Binary masks for each label type
        pos_mask = (targets == 1).float()   # (B, C)
        neg_mask = (targets == 0).float()   # (B, C)
        unc_mask = (targets == -1).float()  # (B, C)

        # ── Positive term: -(1 - p)^y+ * log(p) ─────────────────────────────
        # Focal weight suppresses easy positives (high p) when y+ > 0.
        # y+ = 0 (default) reduces this to standard BCE for positives.
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_pos = torch.pow(1.0 - probs, self.gamma_pos)
        else:
            focal_weight_pos = torch.pow(1.0 - probs, self.gamma_pos)

        pos_term = -focal_weight_pos * torch.log(probs.clamp(min=self.eps))

        # ── Negative term with margin shift: -(p_m)^y- * log(1 - p_m) ───────
        # Margin shift discards near-zero predictions for negatives (easy samples).
        probs_shifted = torch.clamp(probs - self.margin, min=0.0)
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_neg = torch.pow(probs_shifted, self.gamma_neg)
        else:
            focal_weight_neg = torch.pow(probs_shifted, self.gamma_neg)

        neg_term = -focal_weight_neg * torch.log(
            (1.0 - probs_shifted).clamp(min=self.eps)
        )

        # ── Uncertain term: soft blend of positive and negative loss ──────────
        # Uses the unshifted probability to preserve gradient signal for all p.
        # (Giải thích: không dùng margin shift để uncertain labels luôn có gradient,
        #  đặc biệt khi p thấp — margin sẽ zero-out gradient trong trường hợp đó)
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_unc_neg = torch.pow(probs, self.gamma_neg)
        else:
            focal_weight_unc_neg = torch.pow(probs, self.gamma_neg)

        neg_term_no_shift = -focal_weight_unc_neg * torch.log(
            (1.0 - probs).clamp(min=self.eps)
        )

        # Soft interpolation: alpha controls how much uncertain samples lean positive
        unc_term = self.alpha * pos_term + (1.0 - self.alpha) * neg_term_no_shift

        # ── Aggregate losses across confirmed and uncertain samples ───────────
        pos_loss = (pos_term * pos_mask).sum()
        neg_loss = (neg_term * neg_mask).sum()
        unc_loss = self.lambda_unc * (unc_term * unc_mask).sum()
        total_loss = pos_loss + neg_loss + unc_loss

        if self.reduction == "sum":
            return total_loss
        if self.reduction == "none":
            # Return per-element contributions for debugging or custom weighting
            return (
                (pos_term * pos_mask)
                + (neg_term * neg_mask)
                + (self.lambda_unc * unc_term * unc_mask)
            )
        if self.reduction == "mean":
            return total_loss / logits.size(0)

        raise ValueError(f"Unsupported reduction: {self.reduction!r}")