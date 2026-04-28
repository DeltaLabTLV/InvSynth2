"""Unit tests for the InvSynth2 codebase.

Run with:
    pytest tests/

These tests do NOT require GPUs and run quickly. They cover:
  - mask shape and approximate coverage
  - loss numerical sanity (zero loss when pred == target)
  - parameter-loss correctness on continuous + categorical
  - dataset loader file resolution
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from invsynth2.losses.contrastive import nt_xent_loss
from invsynth2.losses.parameter import ParameterLoss
from invsynth2.losses.spectral import (
    IMWLoss,
    LogSpecLoss,
    ReconstructionLoss,
    StandardSpecLoss,
)
from invsynth2.utils.masking import make_frame_mask, make_unet_mask


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def test_frame_mask_shape_and_ratio():
    torch.manual_seed(0)
    mask = make_frame_mask(n_frames=1000, mask_ratio=0.45, batch=4)
    assert mask.shape == (4, 1000)
    # Sample average should be near 0.45.
    assert 0.40 < mask.float().mean().item() < 0.50


def test_unet_mask_coverage():
    torch.manual_seed(0)
    mask = make_unet_mask(n_freq=64, n_time=32, center_ratio=0.065, region_size=3)
    coverage = mask.float().mean().item()
    # Expected ~45% with possible variance; allow 0.30–0.60 band.
    assert 0.30 < coverage < 0.60, f"Coverage out of expected range: {coverage:.3f}"


# --------------------------------------------------------------------------- #
# Spectral losses
# --------------------------------------------------------------------------- #
def test_imw_zero_when_pred_equals_target():
    target = torch.rand(2, 16, 8) * 0.5 + 0.01
    pred = target.clone()
    loss = IMWLoss(epsilon=1e-7)(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_standard_spec_zero_when_pred_equals_target():
    target = torch.rand(2, 16, 8)
    loss = StandardSpecLoss()(target.clone(), target)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_log_spec_zero_when_pred_equals_target():
    target = torch.rand(2, 16, 8) * 0.5 + 0.01
    loss = LogSpecLoss(epsilon=1e-7)(target.clone(), target)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_reconstruction_loss_modes():
    target = torch.rand(2, 16, 8) * 0.5 + 0.01
    pred = target.clone() + 0.01
    for mode in ["imw", "log_spec", "spec_only"]:
        out = ReconstructionLoss(mode=mode)(pred, target)
        assert "loss" in out and out["loss"].item() > 0.0


def test_imw_weights_low_magnitude_more():
    """IMW should penalize the same residual more in low-magnitude regions."""
    # Two single-bin spectrograms: one at magnitude 0.001, one at magnitude 1.0.
    # Both have residual = 0.01.
    low_target = torch.tensor([[[0.001]]])
    high_target = torch.tensor([[[1.0]]])
    low_pred = low_target + 0.01
    high_pred = high_target + 0.01

    imw = IMWLoss(epsilon=1e-7)
    low_loss = imw(low_pred, low_target).item()
    high_loss = imw(high_pred, high_target).item()
    # Loss in the low-magnitude region should be much larger.
    assert low_loss > 100 * high_loss


# --------------------------------------------------------------------------- #
# Parameter loss
# --------------------------------------------------------------------------- #
def test_parameter_loss_continuous_only():
    pred = torch.tensor([[0.5, 0.7, 0.3]])
    gt = torch.tensor([[0.5, 0.7, 0.3]])
    out = ParameterLoss()(pred, gt, pred_cat_logits=[], gt_cat=None)
    assert out["reg"].item() == pytest.approx(0.0, abs=1e-9)
    assert out["cls"].item() == pytest.approx(0.0, abs=1e-9)


def test_parameter_loss_categorical_correct():
    """Confident-correct logits should give near-zero classification loss."""
    pred_cont = torch.zeros(2, 0)
    gt_cont = torch.zeros(2, 0)
    # Strongly confident logits with class 1 correct for both batch elements.
    logits = torch.tensor([[[-10.0, 10.0, -10.0, -10.0]],
                           [[-10.0, 10.0, -10.0, -10.0]]]).squeeze(1)
    gt_cat = torch.tensor([[1], [1]])
    out = ParameterLoss()(pred_cont, gt_cont, pred_cat_logits=[logits], gt_cat=gt_cat)
    assert out["cls"].item() < 1e-3


# --------------------------------------------------------------------------- #
# NT-Xent
# --------------------------------------------------------------------------- #
def test_nt_xent_perfect_alignment_low_loss():
    """When z == positive and negatives are random, NT-Xent should be small."""
    torch.manual_seed(0)
    D = 32
    z = torch.randn(8, D)
    q_pos = z.clone()                  # perfect positives
    q_negs = torch.randn(8, 16, D)     # random negatives
    loss = nt_xent_loss(z, q_pos, q_negs, temperature=0.1)
    assert loss.item() < 1.0    # should be much smaller than a uniform 1+K logit guess


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_dataset_loader_minimal(tmp_path):
    """Smoke test for SynthDataset.__getitem__ with a synthetic 2-sample dataset."""
    import soundfile as sf
    from invsynth2.data.dataset import SynthDataset

    root = tmp_path / "fm"
    (root / "data").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)

    sr = 16_000
    for stem in ["1", "2"]:
        wav = (np.random.randn(sr).astype(np.float32) * 0.1)
        sf.write(str(root / "data" / f"{stem}.wav"), wav, sr)
        np.save(str(root / "labels" / f"{stem}.npy"), np.random.rand(12).astype(np.float32))

    ds = SynthDataset(tmp_path, "fm")
    assert len(ds) == 2
    item = ds[0]
    assert "wav" in item and "theta" in item and "stem" in item
    assert item["wav"].shape == (sr,)
    assert item["theta"].shape == (12,)
