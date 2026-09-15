"""LightningModule for Stage 3: Downstream fine-tuning (synthesizer inversion).

The full pipeline:
    wav → STFT → encoder → pooled features → PEN → θ̂
    θ̂ → frozen proxy P → predicted spectrogram
    L_total = L_rec(P(θ̂), X) + L_reg(θ̂_cont, θ_cont) + L_cls(θ̂_cat, θ_cat)

The ICASSP release exposes only the U-Net path. The exploratory Transformer
implementation remains available in the repository's earlier history.

ITF (Stage 4) is implemented as a separate inference-time function below.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from invsynth2.losses.parameter import ParameterLoss
from invsynth2.losses.spectral import ReconstructionLoss
from invsynth2.models.pen import PEN, pool_unet_features
from invsynth2.models.proxy import IS2Proxy
from invsynth2.models.unet import UNetEncoder
from invsynth2.utils.parameters import (
    ParameterSpec,
    compose_proxy_vector,
    parameter_accuracy,
    split_label,
)
from invsynth2.utils.stft import STFTComputer, STFTConfig, log_to_linear_mag, denormalize_log_mag


class FineTuneModule(L.LightningModule):
    """End-to-end fine-tuning for synthesizer inversion."""

    def __init__(
        self,
        stft_cfg: STFTConfig,
        spec: ParameterSpec,
        encoder_kind: str = "unet",
        loss_mode: str = "imw",                  # "imw" / "log_spec" / "spec_only"
        beta: float = 0.7,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        epsilon: float = 1e-7,
        lambda_param: float = 1.0,
        lambda_rec: float = 1.0,
        lambda_reg: float = 1.0,
        lambda_cls: float = 1.0,
        learning_rate_unet: float = 1e-4,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["stft_cfg", "spec"])
        self.stft_cfg = stft_cfg
        self.spec = spec

        self.stft = STFTComputer(stft_cfg)

        if encoder_kind != "unet":
            raise ValueError("The ICASSP release supports encoder_kind='unet' only")
        self.encoder = UNetEncoder(base_ch=28)
        feature_dim = self.encoder.bottleneck_dim

        self.pen = PEN(in_dim=feature_dim, spec=spec)

        # Proxy is created here but its weights are loaded externally and frozen.
        self.proxy = IS2Proxy(
            n_params=spec.n_total,
            out_freq=stft_cfg.output_freq_bins,
            out_time=stft_cfg.output_frames,
        )
        for p in self.proxy.parameters():
            p.requires_grad = False

        # Losses.
        self.rec_loss = ReconstructionLoss(
            mode=loss_mode, beta=beta, alpha1=alpha1, alpha2=alpha2, epsilon=epsilon,
        )
        self.param_loss = ParameterLoss(lambda_reg=lambda_reg, lambda_cls=lambda_cls)

    # ----------------------------------------------------------------------
    # Loading external weights for encoder + proxy
    # ----------------------------------------------------------------------
    def load_encoder_weights(self, state_dict: dict, strict: bool = True):
        """Load pre-trained encoder weights from a saved Stage-1 checkpoint."""
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=strict)
        print(f"Loaded encoder. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def load_proxy_weights(self, state_dict: dict):
        """Load proxy weights from a saved Stage-2 checkpoint and FREEZE them."""
        self.proxy.load_state_dict(state_dict)
        for p in self.proxy.parameters():
            p.requires_grad = False
        self.proxy.eval()

    # ----------------------------------------------------------------------
    # Forward pipeline
    # ----------------------------------------------------------------------
    def encode(self, log_mag_norm: torch.Tensor) -> torch.Tensor:
        """Run encoder + pooling → (B, D)."""
        pad_f = (-log_mag_norm.shape[-2]) % 16
        pad_t = (-log_mag_norm.shape[-1]) % 16
        padded = torch.nn.functional.pad(log_mag_norm, (0, pad_t, 0, pad_f))
        feats = self.encoder(padded.unsqueeze(1))
        return pool_unet_features(feats["bottleneck"])

    def forward(self, wav: torch.Tensor) -> dict:
        """Inference: wav → θ̂ (continuous + categorical heads)."""
        log_mag_norm = self.stft(wav)["log_mag_norm"]
        feats = self.encode(log_mag_norm)
        return self.pen(feats)

    # ----------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        wav = batch["wav"]
        theta_gt = batch["theta"]                                    # (B, n_total)
        stft_out = self.stft(wav)
        log_mag_norm = stft_out["log_mag_norm"]
        target_lin_mag = stft_out["linear_mag"]                       # for L_rec

        # 1. Encoder + PEN
        feats = self.encode(log_mag_norm)
        head_out = self.pen(feats)                                    # {"cont", "cat_logits"}

        # 2. Compose a differentiable proxy input. Categorical softmax blocks
        # remain relaxed during training and refinement.
        theta_hat = self._compose_theta(head_out, theta_gt)

        # 3. Frozen proxy → predicted spectrogram (in normalized log-mag domain)
        with torch.set_grad_enabled(self.training):
            pred_log_mag_norm = self.proxy(theta_hat)                 # (B, F, T)
            pred_log_mag = denormalize_log_mag(pred_log_mag_norm, self.stft_cfg)
            pred_lin_mag = log_to_linear_mag(pred_log_mag, eps=self.stft_cfg.eps)
            # Match shape with target if proxy's T differs by ±1.
            if pred_lin_mag.shape[-1] != target_lin_mag.shape[-1]:
                pred_lin_mag = nn_resize_time(pred_lin_mag, target_lin_mag.shape[-1])

        # 4. Reconstruction loss on linear magnitude.
        rec = self.rec_loss(pred_lin_mag, target_lin_mag)

        # 5. Parameter regression + classification loss.
        gt_split = split_label(theta_gt, self.spec)
        plosses = self.param_loss(
            head_out["cont"], gt_split["cont"],
            head_out["cat_logits"] if head_out["cat_logits"] else None,
            gt_split["cat"] if gt_split["cat"].numel() > 0 else None,
        )

        # 6. Combine
        loss = self.hparams.lambda_rec * rec["loss"] + self.hparams.lambda_param * plosses["loss"]

        # ----- Logging -----
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/rec_total", rec["loss"], on_epoch=True, sync_dist=True)
        self.log(f"{stage}/rec_spec", rec["spec"], on_epoch=True, sync_dist=True)
        self.log(f"{stage}/rec_aux", rec["aux"], on_epoch=True, sync_dist=True)
        self.log(f"{stage}/p_reg", plosses["reg"], on_epoch=True, sync_dist=True)
        self.log(f"{stage}/p_cls", plosses["cls"], on_epoch=True, sync_dist=True)

        # ACC
        acc = parameter_accuracy(
            head_out["cont"], head_out["cat_logits"], gt_split["cont"], gt_split["cat"], self.spec,
        )
        self.log(f"{stage}/acc", acc, prog_bar=(stage == "val"), on_epoch=True, sync_dist=True)

        return loss

    def _compose_theta(self, head_out: dict, theta_gt: torch.Tensor) -> torch.Tensor:
        """Build the relaxed, proxy-facing vector from PEN outputs."""

        del theta_gt  # kept in the signature for checkpoint/API compatibility
        blocks = [torch.softmax(logits, dim=-1) for logits in head_out["cat_logits"]]
        return compose_proxy_vector(head_out["cont"], blocks, self.spec)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate_unet,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )
        return {
            "optimizer": optimizer,
        }

def nn_resize_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Bilinear resize the time axis of an (B, F, T) magnitude spectrogram."""
    return torch.nn.functional.interpolate(x.unsqueeze(1), size=(x.shape[-2], target_T), mode="bilinear", align_corners=False).squeeze(1)
