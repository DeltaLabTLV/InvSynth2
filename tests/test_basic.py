"""CPU tests for paper-locked configuration, transforms, losses, and decoding."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from invsynth2.data.dataset import ManifestError, load_split_manifest
from invsynth2.losses.spectral import IMWLoss, LogSpecLoss, ReconstructionLoss, StandardSpecLoss
from invsynth2.models.unet import UNetEncoderDecoder
from invsynth2.study import PAPER_CONFIGURATIONS, dataset_profile, load_study_config
from invsynth2.utils.masking import make_unet_mask
from invsynth2.utils.parameters import CategoricalBlock, ParameterSpec, hard_decode_theta, split_label
from invsynth2.utils.stft import STFTComputer, denormalize_log_mag, log_to_linear_mag
from invsynth2.repro import base_feature_config


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "icassp2027.yaml"


def test_study_grid_and_frame_arithmetic():
    cfg = load_study_config(STUDY)
    assert tuple(cfg["configurations"]) == PAPER_CONFIGURATIONS
    expected = {"fm": (257, 129), "dx7": (257, 347), "tal": (257, 129)}
    for name, shape in expected.items():
        profile = dataset_profile(cfg, name)
        assert (profile.computed_freq_bins, profile.computed_frames) == shape


@pytest.mark.parametrize("dataset", ["fm", "dx7", "tal"])
def test_front_end_produces_declared_shape(dataset):
    cfg = load_study_config(STUDY)
    profile = dataset_profile(cfg, dataset)
    feature_cfg = replace(
        base_feature_config(cfg, profile), feature_min=-120.0, feature_max=0.0
    )
    computer = STFTComputer(feature_cfg)
    wav = torch.zeros(1, profile.source_num_samples)
    output = computer(wav)
    assert output["native_mag"].shape[-2:] == (
        profile.expected_freq_bins,
        profile.expected_frames,
    )
    assert output["feature_norm"].dtype == torch.float32


def test_db_affine_and_inverse_are_consistent():
    cfg = load_study_config(STUDY)
    profile = dataset_profile(cfg, "fm")
    feature_cfg = replace(
        base_feature_config(cfg, profile), feature_min=-120.0, feature_max=-4.0
    )
    computer = STFTComputer(feature_cfg)
    db = torch.tensor([-120.0, -62.0, -4.0])
    normalized = computer.normalize_db(db)
    recovered_db = denormalize_log_mag(normalized, feature_cfg)
    assert torch.allclose(recovered_db, db, atol=1e-6)
    assert torch.all(log_to_linear_mag(recovered_db) > 0)


def test_split_manifest_rejects_duplicate_stems(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("stem,split\na,train\na,test\nb,val\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_split_manifest(path)


def test_split_manifest_accepts_disjoint_partitions(tmp_path):
    path = tmp_path / "splits.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows((("stem", "split"), ("a", "train"), ("b", "val"), ("c", "test")))
    assert load_split_manifest(path) == {"train": ["a"], "val": ["b"], "test": ["c"]}


def test_unet_mask_coverage():
    torch.manual_seed(0)
    mask = make_unet_mask(257, 129, center_ratio=0.065, region_size=3)
    assert 0.35 < mask.float().mean().item() < 0.50


def test_unet_parameter_counts_match_manuscript():
    model = UNetEncoderDecoder(base_ch=28)
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder = sum(parameter.numel() for parameter in model.decoder.parameters())
    assert encoder == 3_609_340
    assert decoder == 2_334_529


def test_spectral_losses_are_fp32_and_zero_at_identity():
    target = (torch.rand(2, 16, 8) + 0.01).half()
    for loss in (IMWLoss(), StandardSpecLoss(), LogSpecLoss()):
        value = loss(target.clone(), target)
        assert value.dtype == torch.float32
        assert value.item() == pytest.approx(0.0, abs=1e-8)


def test_reconstruction_modes_have_positive_residual_loss():
    target = torch.rand(2, 16, 8) + 0.01
    prediction = target + 0.01
    for mode in ("imw", "log_spec", "spec_only"):
        result = ReconstructionLoss(mode=mode)(prediction, target)
        assert result["loss"].item() > 0.0


def test_box_relaxation_and_hard_decode_are_distinct():
    spec = ParameterSpec(
        n_total=6,
        cont_indices=(0,),
        categorical_blocks=(
            CategoricalBlock("wave", (1, 2)),
            CategoricalBlock("mode", (3, 4, 5)),
        ),
        name="toy",
    )
    relaxed = torch.tensor([[1.2, 0.4, 0.6, 0.2, 0.7, 0.6]])
    decoded = hard_decode_theta(relaxed, spec)
    assert decoded.tolist() == [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    split = split_label(decoded, spec)
    assert split["cat"].tolist() == [[1, 1]]


def test_reported_aggregate_table_has_complete_4_by_3_by_2_grid():
    path = ROOT / "paper" / "reported_aggregate_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 3 * 2
    assert {row["configuration"] for row in rows} == set(PAPER_CONFIGURATIONS)
    assert {row["dataset"] for row in rows} == {"FM", "DX7", "TAL"}
    assert {row["metric"] for row in rows} == {"Spec", "SC"}
