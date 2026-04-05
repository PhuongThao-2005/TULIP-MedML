import torch
import torch.nn as nn


class UncertaintyAwareASL(nn.Module):
	"""
	Uncertainty-aware Asymmetric Loss for multi-label classification.

	Targets are expected in {-1, 0, 1}:
	  +1 : positive (confirmed)
	   0 : negative (confirmed)
	  -1 : uncertain
	"""

	def __init__(
		self,
		gamma_pos: float = 0.0,
		gamma_neg: float = 4.0,
		margin: float = 0.05,
		lambda_unc: float = 0.5,
		alpha: float = 0.5,
		eps: float = 1e-8,
		reduction: str = 'mean',
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
		if logits.shape != targets.shape:
			raise ValueError(
				f'logits and targets must have the same shape, '
				f'got {tuple(logits.shape)} vs {tuple(targets.shape)}',	
            )

		targets = targets.float()
		p = torch.sigmoid(logits)

		pos_mask = (targets == 1).float()
		neg_mask = (targets == 0).float()
		unc_mask = (targets == -1).float()

        # ── Positive term: -(1-p)^γ+ * log(p) ──────────────────────────────
		if self.disable_torch_grad_focal_loss:
			with torch.no_grad():
				focal_weight_pos = torch.pow(1.0 - p, self.gamma_pos)
		else:
			focal_weight_pos = torch.pow(1.0 - p, self.gamma_pos)
		pos_term = -focal_weight_pos * torch.log(p.clamp(min=self.eps))

        # ── Negative term with margin shift: -(p_m)^γ- * log(1-p_m) ────────
		p_shifted = torch.clamp(p - self.margin, min=0.0)
		if self.disable_torch_grad_focal_loss:
			with torch.no_grad():
				focal_weight_neg = torch.pow(p_shifted, self.gamma_neg)
		else:
			focal_weight_neg = torch.pow(p_shifted, self.gamma_neg)
		neg_term = -focal_weight_neg * torch.log(
			(1.0 - p_shifted).clamp(min=self.eps)
		)

        # ── Uncertain term: soft interpolation, NO margin shift ──────────────
        # Use plain BCE components (no hard-threshold) so uncertain labels
        # always contribute gradient — margin would zero-out low-p uncertain
        # positives, destroying the uncertain signal entirely.
		if self.disable_torch_grad_focal_loss:
			with torch.no_grad():
				focal_weight_unc_neg = torch.pow(p, self.gamma_neg)
		else:
			focal_weight_unc_neg = torch.pow(p, self.gamma_neg)
		neg_term_no_shift = -focal_weight_unc_neg * torch.log(
			(1.0 - p).clamp(min=self.eps)
		)
		# Soft interpolation: uncertain = alpha * pos + (1-alpha) * neg
		unc_term = self.alpha * pos_term + (1.0 - self.alpha) * neg_term_no_shift

        # ── Aggregate ────────────────────────────────────────────────────────
		pos_loss  = (pos_term  * pos_mask).sum()
		neg_loss  = (neg_term  * neg_mask).sum()
		unc_loss  = self.lambda_unc * (unc_term * unc_mask).sum()
		total_loss = pos_loss + neg_loss + unc_loss

		if self.reduction == 'sum':
			return total_loss
		if self.reduction == 'none':
			# Per-element view, useful for debugging.
			return (pos_term * pos_mask) + (neg_term * neg_mask) + (
				self.lambda_unc * unc_term * unc_mask
			)
		elif self.reduction == 'mean':
			return total_loss / logits.size(0)
		else:
			raise ValueError(f'Unsupported reduction: {self.reduction!r}')