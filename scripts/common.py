"""Shared utilities for the entry-point scripts."""

from __future__ import annotations

from pathlib import Path

import yaml
from lightning.pytorch.loggers import TensorBoardLogger


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config and resolve `defaults_from` into the parent dict."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if "defaults_from" in cfg:
        parent_path = config_path.parent / cfg["defaults_from"]
        with parent_path.open("r") as f:
            parent_cfg = yaml.safe_load(f)
        # Merge — child overrides parent.
        merged = _deep_merge(parent_cfg, cfg)
        merged.pop("defaults_from", None)
        return merged
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_tb_logger(
    save_dir: str | Path,
    run_name: str,
) -> TensorBoardLogger:
    """Build a TensorBoardLogger that places logs under save_dir/run_name."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoardLogger(
        save_dir=str(save_dir),
        name=run_name,
        version=None,        # auto-increment
        log_graph=False,
    )
