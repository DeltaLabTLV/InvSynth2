"""LightningModule for Stage 1a: Transformer SSL pre-training (contrastive)."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from invsynth2.losses.contrastive import nt_xent_loss
from invsynth2.models.transformer import TransformerEncoder, make_transformer_encoder
from invsynth2.utils.masking import make_frame_mask
from invsynth2.utils.stft import STFTComputer, STFTConfig


class TransformerPretrainModule(L.LightningModule):
    """Pre-trains the Transformer encoder via masked contrastive learning.

    Per the paper (§3.2): each masked frame z_i is contrasted against its
    positive q_i (encoder output at the same frame on an UNMASKED view, with
    stop-gradient) and within-spectrogram negative distractors.
    """

    def __init__(
        self,
        stft_cfg: STFTConfig,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-6,
        mask_ratio: float = 0.45,
        n_negatives: int = 50,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["stft_cfg"])
        self.stft_cfg = stft_cfg

        self.stft = STFTComputer(stft_cfg)
        self.encoder: TransformerEncoder = make_transformer_encoder()

    # ---- training/validation step ----------------------------------------
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        wav = batch["wav"]                              # (B, T_audio)
        log_mag_norm = self.stft(wav)["log_mag_norm"]   # (B, F, T)
        B, _F, T = log_mag_norm.shape

        # 1. Forward UNMASKED view (with stop-gradient afterwards) → q targets.
        with torch.no_grad():
            unmasked_tokens = self.encoder(log_mag_norm, frame_mask=None)  # (B, T, D)

        # 2. Forward MASKED view → z embeddings at masked positions.
        frame_mask = make_frame_mask(
            T, mask_ratio=self.hparams.mask_ratio, batch=B, device=log_mag_norm.device
        )
        masked_tokens = self.encoder(log_mag_norm, frame_mask=frame_mask)   # (B, T, D)

        # 3. Build (z_masked, q_pos, q_negs) tuples.
        z_list, qpos_list, qneg_list = [], [], []
        for b in range(B):
            mask_b = frame_mask[b]                        # (T,) bool
            masked_idx = mask_b.nonzero(as_tuple=False).squeeze(-1)
            unmasked_idx = (~mask_b).nonzero(as_tuple=False).squeeze(-1)
            if masked_idx.numel() == 0 or unmasked_idx.numel() == 0:
                continue
            z = masked_tokens[b, masked_idx]                # (M_b, D)
            q_pos = unmasked_tokens[b, masked_idx].detach() # (M_b, D)
            # Sample K negatives per masked position from unmasked frames.
            K = self.hparams.n_negatives
            if unmasked_idx.numel() >= K:
                neg_idx = unmasked_idx[
                    torch.randperm(unmasked_idx.numel(), device=z.device)[:K]
                ].unsqueeze(0).expand(masked_idx.numel(), -1)
            else:
                neg_idx = unmasked_idx[
                    torch.randint(0, unmasked_idx.numel(), (masked_idx.numel(), K), device=z.device)
                ]
            q_negs = unmasked_tokens[b][neg_idx].detach()   # (M_b, K, D)
            z_list.append(z)
            qpos_list.append(q_pos)
            qneg_list.append(q_negs)

        if not z_list:  # all-masked or all-unmasked degenerate batch
            loss = log_mag_norm.sum() * 0.0
        else:
            z_masked = torch.cat(z_list, dim=0)
            q_pos = torch.cat(qpos_list, dim=0)
            q_negs = torch.cat(qneg_list, dim=0)
            loss = nt_xent_loss(z_masked, q_pos, q_negs, temperature=self.hparams.temperature)

        self.log(f"{stage}/nt_xent", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )
        return opt

    # Convenience: extract just the encoder state dict for downstream use.
    def get_encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()
