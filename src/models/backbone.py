import torch
import torch.nn as nn
import timm


class SwinTBackbone(nn.Module):
	def __init__(self, pretrained: bool = True):
		super().__init__()
		self.model = timm.create_model(
			"swin_tiny_patch4_window7_224",
			pretrained=pretrained,
			num_classes=0,
			global_pool="",
		)

		self.token_pool = nn.AdaptiveAvgPool1d(1)
		self.proj = nn.Linear(768, 2048)

		self.image_normalization_mean = [0.485, 0.456, 0.406]
		self.image_normalization_std = [0.229, 0.224, 0.225]

	def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
		# Convert feature maps to token format (B, N, C) before token pooling.
		if x.dim() == 3:
			return x

		if x.dim() == 4:
			if x.shape[-1] == 768:
				b, h, w, c = x.shape
				return x.view(b, h * w, c)

			b, c, h, w = x.shape
			return x.flatten(2).transpose(1, 2)

		raise ValueError(f"Unexpected Swin output shape: {tuple(x.shape)}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		tokens = self._to_tokens(self.model(x))
		pooled = self.token_pool(tokens.transpose(1, 2)).squeeze(-1)
		out = self.proj(pooled)
		return out
