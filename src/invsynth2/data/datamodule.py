"""Lightning DataModule with deterministic 80/10/10 split.

The split is computed once from `numpy.random.default_rng(seed)` and is
identical across all stages of training (pre-train / proxy / fine-tune)
because the same seed is used in every stage's config.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from invsynth2.data.dataset import SynthDataset


class SynthDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        target_seconds: float = 1.0,
        sample_rate: int = 16_000,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root)
        self.dataset_name = dataset_name

        self.train_set: SynthDataset | None = None
        self.val_set: SynthDataset | None = None
        self.test_set: SynthDataset | None = None

    def setup(self, stage: str | None = None):
        # Discover total file count by instantiating a probe dataset.
        probe = SynthDataset(
            self.root, self.dataset_name,
            target_seconds=self.hparams.target_seconds,
            sample_rate=self.hparams.sample_rate,
        )
        n_total = len(probe)
        # Deterministic split — re-seeded each call for reproducibility.
        rng = np.random.default_rng(self.hparams.seed)
        perm = rng.permutation(n_total)

        n_train = int(round(self.hparams.train_frac * n_total))
        n_val = int(round(self.hparams.val_frac * n_total))
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train : n_train + n_val].tolist()
        test_idx = perm[n_train + n_val :].tolist()

        self.train_set = SynthDataset(
            self.root, self.dataset_name, sample_indices=train_idx,
            target_seconds=self.hparams.target_seconds,
            sample_rate=self.hparams.sample_rate,
        )
        self.val_set = SynthDataset(
            self.root, self.dataset_name, sample_indices=val_idx,
            target_seconds=self.hparams.target_seconds,
            sample_rate=self.hparams.sample_rate,
        )
        self.test_set = SynthDataset(
            self.root, self.dataset_name, sample_indices=test_idx,
            target_seconds=self.hparams.target_seconds,
            sample_rate=self.hparams.sample_rate,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
