"""LightningModule for Stage 2: Proxy training.

Trains the IS2-style differentiable proxy P:  θ → log-magnitude spectrogram.
The proxy is used by the downstream fine-tuning stage to compute the
reconstruction loss; it is frozen during fine-tuning and ITF.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from invsynth2.models.proxy import IS2Proxy
from invsynth2.utils.parameters import ParameterSpec
from invsynth2.utils.stft import STFTComputer, STFTConfig


class ProxyTrainModule(L.LightningModule):
    """Supervised training of the proxy: minimize MSE(log|P(θ)|, log|X|).

    Uses log-magnitude target so the proxy learns the same representation that
    the encoder consumes. After training, the proxy is loaded into the
    fine-tuning module with `requires_grad=False`.
    """

    def __init__(
        self,
        stft_cfg: STFTConfig,
        spec: ParameterSpec,
        out_freq: int = 513,
        out_time: int = 63,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["stft_cfg", "spec"])
        self.stft_cfg = stft_cfg
        self.spec = spec

        self.stft = STFTComputer(stft_cfg)
        self.proxy = IS2Proxy(
            n_params=spec.n_total,
            out_freq=out_freq,
            out_time=out_time,
        )

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        wav = batch["wav"]
        theta = batch["theta"]                              # (B, n_total)
        log_mag_norm = self.stft(wav)["log_mag_norm"]       # (B, F, T)

        pred_log_mag_norm = self.proxy(theta)               # (B, F, T)
        loss = F.mse_loss(pred_log_mag_norm, log_mag_norm)
        self.log(f"{stage}/proxy_mse", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
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
