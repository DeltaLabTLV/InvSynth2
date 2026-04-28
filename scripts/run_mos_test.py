"""Generate audio stimuli for a MOS listening test.

For each (dataset, system, sample) tuple we save:
  - target.wav         the ground-truth audio
  - reconstruction.wav the system's reconstruction

We also emit `manifest.csv` that listeners' rating tools should consume,
listing `stimulus_id`, the dataset, the system, and the audio paths.

Note: this script writes the *target* audio from the test set and a
*placeholder* reconstruction WAV produced by Griffin-Lim from the system's
predicted spectrogram. The paper itself collected baseline (Flow / IS / IS2 /
IS2xITF) audio outputs from the IS2 authors (cf. §4.3); for our two systems we
generate the audio here.

Usage:
    python scripts/run_mos_test.py \
        --output-dir mos_stimuli/ \
        --datasets fm dx7 talnoise \
        --systems transformer unet \
        --transformer-ckpt runs/finetune_transformer_fm_imw_seed42/ckpts/best.ckpt \
        --unet-ckpt        runs/finetune_unet_fm_imw_seed42/ckpts/best.ckpt \
        --n-stimuli 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lightning as L
import numpy as np
import soundfile as sf
import torch

from invsynth2.data import SynthDataModule
from invsynth2.training import FineTuneModule
from invsynth2.utils.parameters import DEFAULT_SPECS
from invsynth2.utils.stft import denormalize_log_mag, log_to_linear_mag


def _spec_for(dataset: str):
    if dataset in DEFAULT_SPECS:
        return DEFAULT_SPECS[dataset]
    if dataset == "talnoise":
        return DEFAULT_SPECS["tal"]
    raise ValueError(f"No ParameterSpec for dataset={dataset!r}")


def griffin_lim(
    mag: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_iter: int = 32,
) -> torch.Tensor:
    """Simple Griffin–Lim phase reconstruction for sanity-check audio playback.

    For perceptual evaluation, prefer the synthesizer's actual rendered audio.
    This is a fallback used only when we can't re-synthesize from θ̂ directly.
    """
    window = torch.hann_window(win_length, device=mag.device)
    angles = torch.empty_like(mag).uniform_(-1.0, 1.0) * np.pi
    spec = mag * torch.exp(1j * angles)
    for _ in range(n_iter):
        wav = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        new_spec = torch.stft(
            wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True,
        )
        spec = mag * (new_spec / new_spec.abs().clamp(min=1e-8))
    wav = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    return wav


def reconstruct_audio_from_module(
    module: FineTuneModule, wav_target: torch.Tensor,
) -> torch.Tensor:
    """Push the target through encoder→PEN→proxy and Griffin–Lim it back to audio."""
    module.eval()
    with torch.no_grad():
        log_mag_norm = module.stft(wav_target)["log_mag_norm"]
        feats = module.encode(log_mag_norm)
        head_out = module.pen(feats)
        theta_hat = module._compose_theta(head_out, wav_target.new_zeros(wav_target.shape[0], module.spec.n_total))
        pred_log_mag_norm = module.proxy(theta_hat)
        pred_log_mag = denormalize_log_mag(pred_log_mag_norm, module.stft_cfg)
        pred_lin = log_to_linear_mag(pred_log_mag, eps=module.stft_cfg.eps)

        # Match length axis with target if proxy time-resolution differs by one.
        target_lin = module.stft(wav_target)["linear_mag"]
        if pred_lin.shape[-1] != target_lin.shape[-1]:
            pred_lin = torch.nn.functional.interpolate(
                pred_lin.unsqueeze(1),
                size=(pred_lin.shape[-2], target_lin.shape[-1]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        wav_recon = griffin_lim(
            pred_lin[0],
            n_fft=module.stft_cfg.n_fft,
            hop_length=module.stft_cfg.hop_length,
            win_length=module.stft_cfg.win_length,
        )
    return wav_recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--systems", type=str, nargs="+", default=["transformer", "unet"],
                        choices=["transformer", "unet"])
    parser.add_argument("--transformer-ckpt", type=str, default=None)
    parser.add_argument("--unet-ckpt", type=str, default=None)
    parser.add_argument("--n-stimuli", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV manifest
    manifest_rows = []
    csv_path = out / "manifest.csv"

    for dataset in args.datasets:
        spec = _spec_for(dataset)
        # Load each requested system module.
        modules: dict[str, FineTuneModule] = {}
        if "transformer" in args.systems:
            if not args.transformer_ckpt:
                raise ValueError("--transformer-ckpt required when 'transformer' is in --systems")
            modules["transformer"] = FineTuneModule.load_from_checkpoint(
                args.transformer_ckpt, spec=spec,
            )
        if "unet" in args.systems:
            if not args.unet_ckpt:
                raise ValueError("--unet-ckpt required when 'unet' is in --systems")
            modules["unet"] = FineTuneModule.load_from_checkpoint(args.unet_ckpt, spec=spec)
        for m in modules.values():
            m.eval()

        # Pull stimuli from the test split (same seeded split as training).
        dm = SynthDataModule(
            root=args.data_root, dataset_name=dataset,
            batch_size=1, num_workers=0, seed=args.seed,
        )
        dm.setup()
        loader = dm.test_dataloader()

        for i, batch in enumerate(loader):
            if i >= args.n_stimuli:
                break
            stem = batch["stem"][0]
            wav_target = batch["wav"]               # (1, T)
            audio_np = wav_target[0].cpu().numpy()
            stim_dir = out / dataset / stem
            stim_dir.mkdir(parents=True, exist_ok=True)

            target_path = stim_dir / "target.wav"
            sf.write(str(target_path), audio_np, args.sample_rate)
            manifest_rows.append({
                "stimulus_id": f"{dataset}_{stem}",
                "dataset": dataset,
                "system": "target",
                "audio_path": str(target_path.relative_to(out)),
            })
            for sys_name, mod in modules.items():
                recon = reconstruct_audio_from_module(mod, wav_target.to(next(mod.parameters()).device))
                # Resize to fixed length to match target.
                recon_np = recon.cpu().numpy().astype(np.float32)
                # RMS-normalize for fair playback.
                rms = float(np.sqrt(np.mean(recon_np**2) + 1e-9))
                if rms > 1e-8:
                    recon_np = recon_np * (0.1 / rms)
                if len(recon_np) > len(audio_np):
                    recon_np = recon_np[: len(audio_np)]
                elif len(recon_np) < len(audio_np):
                    recon_np = np.pad(recon_np, (0, len(audio_np) - len(recon_np)))
                recon_path = stim_dir / f"{sys_name}.wav"
                sf.write(str(recon_path), recon_np, args.sample_rate)
                manifest_rows.append({
                    "stimulus_id": f"{dataset}_{stem}",
                    "dataset": dataset,
                    "system": sys_name,
                    "audio_path": str(recon_path.relative_to(out)),
                })

    # Write manifest
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stimulus_id", "dataset", "system", "audio_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} rows to {csv_path}")
    print(f"Audio stimuli are under {out}")
    print("\nNext step: distribute the manifest + audio to listeners and collect a ratings CSV with columns:")
    print("    listener_id, dataset, system, stimulus_id, rating  (rating in {1,2,3,4,5})")
    print("Then run `scripts/compute_mos.py --ratings ratings.csv`.")


if __name__ == "__main__":
    main()
