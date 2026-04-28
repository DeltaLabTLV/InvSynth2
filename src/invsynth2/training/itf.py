"""Stage 4: Inference-Time Fine-tuning (ITF).

At inference, given a target spectrogram X, we refine the predicted parameter
vector θ̂ by minimizing the spectral reconstruction loss through the frozen
proxy P. The encoder is also frozen — only θ̂ is updated.

This is implemented as a function rather than a LightningModule because it is
a per-sample optimization run during evaluation, not a training stage.
"""

from __future__ import annotations

import torch

from invsynth2.losses.spectral import ReconstructionLoss
from invsynth2.training.finetune import FineTuneModule, nn_resize_time
from invsynth2.utils.parameters import split_label
from invsynth2.utils.stft import denormalize_log_mag, log_to_linear_mag


@torch.no_grad()
def _initial_theta_hat(module: FineTuneModule, wav: torch.Tensor) -> torch.Tensor:
    """Get the encoder+PEN's first guess for θ̂."""
    log_mag_norm = module.stft(wav)["log_mag_norm"]
    feats = module.encode(log_mag_norm)
    head_out = module.pen(feats)
    return module._compose_theta(head_out, wav.new_zeros(wav.shape[0], module.spec.n_total))


def itf_refine(
    module: FineTuneModule,
    wav: torch.Tensor,
    n_steps: int = 100,
    lr: float = 1e-2,
    loss_mode: str | None = None,
    beta: float | None = None,
    verbose: bool = False,
) -> dict:
    """Refine θ̂ at inference time via gradient descent through the frozen proxy.

    Encoder and proxy weights are frozen. Only the leaf tensor θ̂ is optimized.

    Parameters
    ----------
    module: a fine-tuned FineTuneModule.
    wav: (B, T_audio) input waveform(s) to invert.
    n_steps: number of optimization steps.
    lr: SGD step size on θ̂.
    loss_mode/beta: override the reconstruction loss config (defaults to
                    whatever was used during fine-tuning).

    Returns
    -------
    {"theta_hat_initial": (B, n_total),
     "theta_hat_refined": (B, n_total),
     "loss_history": list of floats per step}
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

    # Compute the target spectrogram once.
    stft_out = module.stft(wav)
    target_lin_mag = stft_out["linear_mag"].detach()

    # Initialize θ̂ from the encoder's prediction.
    theta_hat_init = _initial_theta_hat(module, wav).detach()
    theta_hat = theta_hat_init.clone().requires_grad_(True)

    # Build the reconstruction loss (use module's default if not overridden).
    rec_mode = loss_mode or module.hparams.loss_mode
    rec_beta = beta if beta is not None else module.hparams.beta
    rec_loss = ReconstructionLoss(
        mode=rec_mode, beta=rec_beta,
        alpha1=module.hparams.alpha1, alpha2=module.hparams.alpha2,
        epsilon=module.hparams.epsilon,
    ).to(wav.device)

    optimizer = torch.optim.Adam([theta_hat], lr=lr)
    history: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()
        # Clamp to [0, 1] so values stay in valid synthesizer-parameter range
        # without affecting gradients on in-bounds entries.
        theta_clamped = theta_hat.clamp(min=0.0, max=1.0)
        pred_log_mag_norm = module.proxy(theta_clamped)
        pred_log_mag = denormalize_log_mag(pred_log_mag_norm, module.stft_cfg)
        pred_lin_mag = log_to_linear_mag(pred_log_mag, eps=module.stft_cfg.eps)
        if pred_lin_mag.shape[-1] != target_lin_mag.shape[-1]:
            pred_lin_mag = nn_resize_time(pred_lin_mag, target_lin_mag.shape[-1])
        loss = rec_loss(pred_lin_mag, target_lin_mag)["loss"]
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"[ITF step {step:03d}] loss = {loss.item():.6f}")

    return {
        "theta_hat_initial": theta_hat_init,
        "theta_hat_refined": theta_hat.detach().clamp(0.0, 1.0),
        "loss_history": history,
    }
