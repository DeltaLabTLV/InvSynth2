"""U-Net for MAE pre-training.

§3.2 of the paper:
- Encoder + decoder pre-trained jointly to reconstruct masked spectrogram regions.
- After pre-training, the decoder is discarded — only the encoder is retained
  for downstream inversion.
- Encoder ≈ 3.5M params. Skip connections support multi-resolution recovery.

Operates on log-magnitude spectrograms shaped (B, 1, F, T). For F=513, T=63
the model has 4 down/up levels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        # GroupNorm groups must divide channel count. Pick the largest power-of-2
        # divisor up to 8 to keep numerics stable while supporting non-multiples-of-8.
        groups = 1
        for g in (8, 4, 2):
            if out_ch % g == 0:
                groups = g
                break
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd-sized skip from non-power-of-two F dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    """The encoder half of the U-Net. ~3.5M parameters at default settings.

    Returns a feature map (B, C, F', T') and the intermediate skip activations
    used by the decoder during pre-training.
    """

    def __init__(self, base_ch: int = 28):
        super().__init__()
        self.in_conv = ConvBlock(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        self.bottleneck_dim = base_ch * 16

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, 1, F, T)
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        z = self.down4(s3)
        return {"bottleneck": z, "skips": [s0, s1, s2, s3]}


class UNetDecoder(nn.Module):
    """Decoder used only during MAE pre-training to reconstruct the masked input."""

    def __init__(self, base_ch: int = 28):
        super().__init__()
        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s0, s1, s2, s3 = skips
        x = self.up1(z, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.out_conv(x)


class UNetEncoderDecoder(nn.Module):
    """Wraps encoder + decoder for MAE pre-training. After training, save the
    encoder's state_dict separately and discard the decoder."""

    def __init__(self, base_ch: int = 28):
        super().__init__()
        self.encoder = UNetEncoder(base_ch=base_ch)
        self.decoder = UNetDecoder(base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats["bottleneck"], feats["skips"])


def make_unet(target_params: float = 3.5e6) -> UNetEncoderDecoder:
    """Build a U-Net whose encoder is ≈3.5M parameters at the default base_ch=28."""
    return UNetEncoderDecoder(base_ch=28)
