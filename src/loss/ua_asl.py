import torch
import torch.nn as nn


class UncertaintyAwareASL(nn.Module):
	"""
	Uncertainty-aware Asymmetric Loss for multi-label classification.

	Targets are expected in {-1, 0, 1}:
	  -  1: positive label
	  -  0: negative label
	  - -1: uncertain label
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
	):
		super().__init__()
		self.gamma_pos = gamma_pos
		self.gamma_neg = gamma_neg
		self.margin = margin
		self.lambda_unc = lambda_unc
		self.alpha = alpha
		self.eps = eps
		self.reduction = reduction

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		if logits.shape != targets.shape:
			raise ValueError(
				f'logits and targets must have the same shape, got '
				f'{tuple(logits.shape)} vs {tuple(targets.shape)}'
			)

		targets = targets.float()
		p = torch.sigmoid(logits)

		pos_mask = (targets == 1).float()
		neg_mask = (targets == 0).float()
		unc_mask = (targets == -1).float()

		# Positive loss: - (1 - p)^gamma_pos * log(p)
		pos_term = -torch.pow(1.0 - p, self.gamma_pos) * torch.log(p.clamp(min=self.eps))

		# Negative loss with margin shift:
		# p_shifted = max(p - margin, 0)
		# - (p_shifted)^gamma_neg * log(1 - p_shifted)
		p_shifted = torch.clamp(p - self.margin, min=0.0)
		neg_term = -torch.pow(p_shifted, self.gamma_neg) * torch.log(
			(1.0 - p_shifted).clamp(min=self.eps)
		)

		pos_loss = (pos_term * pos_mask).sum()
		neg_loss = (neg_term * neg_mask).sum()

		# Soft uncertain target: lambda_unc * (alpha * L_pos + (1-alpha) * L_neg)
		unc_term = self.alpha * pos_term + (1.0 - self.alpha) * neg_term
		unc_loss = self.lambda_unc * (unc_term * unc_mask).sum()

		total_loss = pos_loss + neg_loss + unc_loss

		if self.reduction == 'sum':
			return total_loss
		if self.reduction == 'none':
			# Per-element view, useful for debugging.
			return (pos_term * pos_mask) + (neg_term * neg_mask) + (
				self.lambda_unc * unc_term * unc_mask
			)
		if self.reduction == 'mean':
			denom = (pos_mask + neg_mask + unc_mask).sum().clamp(min=1.0)
			return total_loss / denom

		raise ValueError(f'Unsupported reduction: {self.reduction}')
