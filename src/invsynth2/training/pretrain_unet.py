"""LightningModule for Stage 1b: U-Net MAE pre-training.

Per the paper (§3.2): rectangular pixel masks (~45% coverage from 6.5% centers
+ 3x3 region) are placed on the spectrogram; the encoder + decoder are jointly
trained to reconstruct the full spectrogram from the visible context. Only the
encoder is retained for downstream inversion.
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

        # Forward through the encoder-decoder.
        x_in = masked_input.unsqueeze(1)                 # (B, 1, F, T)
        recon = self.model(x_in).squeeze(1)              # (B, F, T)

        # Loss is MSE on MASKED bins only — standard MAE objective.
        loss_per_bin = (recon - log_mag_norm) ** 2
        n_masked = mask.float().sum().clamp(min=1.0)
        loss = (loss_per_bin * mask.float()).sum() / n_masked

        self.log(f"{stage}/mae_recon", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def get_encoder_state_dict(self) -> dict:
        """Encoder-only state dict for downstream fine-tuning."""
        return self.model.encoder.state_dict()
