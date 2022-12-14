"""
Define a type schema for configuration files for this project.
Source: https://omegaconf.readthedocs.io/en/latest/structured_config.html

Avoid setting defaults, except None for optional values.

Created: Dec 2022 by Eric DeMattos
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf, ValidationError, open_dict

logger = logging.getLogger(__name__)


@dataclass
class LSTM:
    hidden_size: int
    num_layers: int
    bidirectional: Optional[bool] = field(default=False)  # PyTorch default
    dropout: Optional[float] = field(default=0.0)  # PyTorch default


@dataclass
class Arch:
    lstm: LSTM


@dataclass
class CUDA:
    num_gpus: int
    visible_devices: Optional[list[int]] = field(
        default_factory=lambda: [i for i in range(torch.cuda.device_count())]
    )


@dataclass
class Data:
    x_dir: Path
    y_dir: Path


@dataclass
class Test:
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
    cuda: CUDA
    experiment: str
    num_workers: int
    repo_dir: Path
    train: Train
    seed: Optional[int] = None
    test: Optional[Test] = None


def validate_device(cfg: DictConfig) -> DictConfig:

    if cfg.cuda.num_gpus != 0 and not torch.cuda.is_available():

        logger.warning("GPU requested but not available on this machine")
        cfg.cuda.num_gpus = 0

    elif cfg.cuda.num_gpus != 0:

        total_gpus = torch.cuda.device_count()

        if cfg.cuda.num_gpus < 0:
            with open_dict(cfg):
                cfg.cuda.num_gpus = total_gpus

        log = f"Requested {cfg.cuda.num_gpus} of {total_gpus} available GPUs"

        if cfg.cuda.num_gpus > total_gpus:
            logger.warning(log)
            logger.warning(f"Resuming with {total_gpus} GPU")
            cfg.cuda.num_gpus = total_gpus
        else:
            logger.info(log)

    if cfg.cuda.num_gpus == 0:
        logger.error("Trainer assumes GPU-based training")
        logger.error("To use CPU, remove DDP and mixed precision")
        exit(1)

    cuda = f"cuda (num_gpus={cfg.cuda.num_gpus})"
    logger.info(f"Using {cuda if cfg.cuda.num_gpus > 0 else 'cpu'}")

    return cfg


def validate_config(cfg: DictConfig) -> DictConfig:
    """
    Sanity check the config with type hints, and dynamically configure
    settings if necessary (ex. GPU availability).

    OmegaConf handles validation lazily: an exception is raised only when an
    invalid attribute is accessed. If a mandatory field is missing but never
    accessed, no error is thrown.

    Error message will indicate whether it was a type mismatch, missing
    mandatory field, etc.
    """
    cfg = OmegaConf.merge(OmegaConf.structured(Schema), cfg)

    # ensure GPU(s) is/are available
    cfg = validate_device(cfg)

    # make sure path/to/repo exists for relative interpolated paths to work
    if not cfg.repo_dir.exists():
        raise ValidationError(f"Could not locate repo at: {cfg.repo_dir}")

    return cfg
