"""Parameter Estimation Network (PEN).

Maps encoder features → predicted synthesizer parameters θ̂ with separate output
branches for continuous (regression) and categorical (classification) parameter
groups, following the IS2 head structure.

The PEN is encoder-agnostic: it accepts pooled features of shape (B, D_in).
Pooling strategy is delegated to the caller (mean over time for Transformer,
global avg pool over (F, T) for U-Net).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from invsynth2.utils.parameters import ParameterSpec


class PEN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        spec: ParameterSpec,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spec = spec
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cont_head = (
            nn.Linear(hidden_dim, spec.n_continuous) if spec.n_continuous > 0 else None
        )
        # One classification head per categorical parameter (each has its own num_classes).
        self.cat_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for (_, num_classes) in spec.cat_specs]
        )

    def forward(self, features: torch.Tensor) -> dict:
        h = self.trunk(features)
        out: dict = {"cat_logits": []}
        if self.cont_head is not None:
            # Continuous parameters are normalized to [0, 1] with a sigmoid.
            out["cont"] = torch.sigmoid(self.cont_head(h))
        else:
            out["cont"] = features.new_zeros(h.shape[0], 0)
        for head in self.cat_heads:
            out["cat_logits"].append(head(h))
        return out


def pool_transformer_features(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool Transformer (B, T, D) features into (B, D)."""
    return tokens.mean(dim=1)


def pool_unet_features(feats: torch.Tensor) -> torch.Tensor:
    """Global avg-pool U-Net bottleneck (B, C, F', T') into (B, C)."""
    return feats.mean(dim=(-1, -2))
