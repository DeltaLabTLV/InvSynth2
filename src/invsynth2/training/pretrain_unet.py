"""LightningModule for Stage 1b: U-Net MAE pre-training.

Rectangular pixel masks (~45% coverage from 6.5% centers + 3x3 regions)
are reconstructed only on their union. Inputs are padded to multiples of 16
before four pooling levels and cropped back before the masked reduction.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from invsynth2.models.unet import UNetEncoderDecoder, make_unet
from invsynth2.utils.masking import make_unet_mask
from invsynth2.utils.stft import STFTComputer, STFTConfig


class UNetPretrainModule(L.LightningModule):
    def __init__(
        self,
        stft_cfg: STFTConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        mask_center_ratio: float = 0.065,
        mask_region_size: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["stft_cfg"])
        self.stft_cfg = stft_cfg

        self.stft = STFTComputer(stft_cfg)
        self.model: UNetEncoderDecoder = make_unet()

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        wav = batch["wav"]
        log_mag_norm = self.stft(wav)["log_mag_norm"]   # (B, F, T)
        B, F_, T = log_mag_norm.shape

        # Build rectangular pixel mask: True = MASKED.
        mask = make_unet_mask(
            n_freq=F_, n_time=T,
            center_ratio=self.hparams.mask_center_ratio,
            region_size=self.hparams.mask_region_size,
            batch=B, device=log_mag_norm.device,
        )

        # Replace masked regions with 0 (already-normalized => mean of distribution).
        masked_input = log_mag_norm.clone()
        masked_input[mask] = 0.0

        # Paper-locked U-Net support: pad high-frequency/trailing-time edges,
        # pass four pooling levels, and crop before evaluating the loss.
        pad_f = (-F_) % 16
        pad_t = (-T) % 16
        x_in = F.pad(masked_input, (0, pad_t, 0, pad_f)).unsqueeze(1)
        recon = self.model(x_in).squeeze(1)[..., :F_, :T]

        # Loss is MSE on MASKED bins only — standard MAE objective.
        with torch.autocast(device_type=recon.device.type, enabled=False):
            loss_per_bin = (recon.float() - log_mag_norm.float()).square()
            mask32 = mask.float()
            n_masked = mask32.sum().clamp(min=1.0)
            loss = (loss_per_bin * mask32).sum() / n_masked

        self.log(f"{stage}/mae_recon", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def get_encoder_state_dict(self) -> dict:
        """Encoder-only state dict for downstream fine-tuning."""
        return self.model.encoder.state_dict()
