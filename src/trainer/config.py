"""
Define a type schema for configuration files for this project.
Source: https://omegaconf.readthedocs.io/en/latest/structured_config.html

Avoid setting defaults, except None for optional values.

Created: Dec 2022 by Eric DeMattos
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf, ValidationError, open_dict

logger = logging.getLogger(__name__)


@dataclass
class LSTM:
    hidden_size: int
    num_layers: int
    bidirectional: Optional[bool] = None
    dropout: Optional[float] = None


@dataclass
class Arch:
    lstm: LSTM


@dataclass
class CUDA:
    device: int
    num_gpus: int
    visible_devices: Optional[list[int]] = None


@dataclass
class Data:
    x_dir: Path
    y_dir: Path


@dataclass
class Test:
    batch_size: int
    data: Data


@dataclass
class Train:
    batch_size: int
    epochs: int
    data: Data
    lr: float


@dataclass
class Schema:
    arch: Arch
    checkpoint_dir: Path
    experiment: str
    num_workers: int
    repo_dir: Path
    train: Train
    cuda: Optional[CUDA] = None
    seed: Optional[int] = None
    test: Optional[Test] = None
    verbose: Optional[bool] = None


def validate_device(cfg: DictConfig) -> DictConfig:

    if cfg.cuda.num_gpus != 0 and not torch.cuda.is_available():

        logger.warning("GPU requested but not available on this machine")
        cfg.cuda.num_gpus = 0

    elif cfg.cuda.num_gpus != 0:

        total_gpus = torch.cuda.device_count()

        if cfg.cuda.num_gpus == -1:
            with open_dict(cfg):
                cfg.cuda.num_gpus = total_gpus

        logger.info(f"Requested {cfg.cuda.num_gpus} of {total_gpus} GPUs")

    cuda = f"cuda (num_gpus={cfg.cuda.num_gpus})"
    logger.info(f"Using {cuda if cfg.cuda.num_gpus > 0 else 'cpu'}")

    with open_dict(cfg):
        cfg.device = "cuda" if cfg.cuda.num_gpus > 0 else "cpu"

    return cfg


def type_validate_config(cfg: DictConfig) -> DictConfig:
    """
    Sanity check the config with type hints, and dynamically configure
    settings if necessary (ex. GPU availability).
    """
    logger.debug("Validating configuration file")
    try:
        cfg = OmegaConf.merge(OmegaConf.structured(Schema), cfg)
    except ValidationError as e:
        logger.error(e)
        exit(1)

    # ensure GPU(s) is/are available
    cfg = validate_device(cfg)

    # make sure path/to/repo exists for relative interpolated paths to work
    if not cfg.repo_dir.exists():
        raise ValidationError(f"Could not locate repo at: {cfg.repo_dir}")

    return cfg
