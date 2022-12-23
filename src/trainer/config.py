"""
Define a type schema for configuration files for this project.
Source: https://omegaconf.readthedocs.io/en/latest/structured_config.html

Created: Dec 2022 by Eric DeMattos
"""
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class LSTM:
    bidirectional: bool
    dropout: float
    hidden_size: int
    num_layers: int


@dataclass
class Arch:
    lstm: LSTM


@dataclass
class CUDA:
    num_gpus: int  # negative number resolves to max number of GPUs available
    visible_devices: Optional[list[int]] = field(
        default_factory=lambda: [i for i in range(torch.cuda.device_count())]
    )  # expose all GPU devices by default


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
    cuda: CUDA
    experiment: str
    experiment_dir: Path
    log: Path
    num_workers: int
    repo_dir: Path
    train: Train
    overwrite: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    test: Optional[Test] = field(default=None)


def _convert_path_to_str(container: dict[str, Any]) -> dict[str, Any]:
    """
    OmegaConf serializes Path objects in a non-straightforward way
    instead of just converting them to simple strings.
    """
    for key, value in container.items():
        if isinstance(value, dict):
            _convert_path_to_str(value)
        elif isinstance(value, Path):
            container[key] = str(value)
    return container


def _pop(node):
    """
    Get first element in a list. https://stackoverflow.com/a/71043429/9998470
    """
    return node[0]


def configure_device(cfg: DictConfig) -> DictConfig:
    """
    Resolves (and potentially updates) cfg.cuda.num_gpus based on availability.
    """

    if cfg.cuda.num_gpus != 0 and not torch.cuda.is_available():

        logger.warning("GPU requested but not available on this machine")
        OmegaConf.update(cfg, "cuda.num_gpus", 0)

    elif cfg.cuda.num_gpus != 0:

        # determine the maximum amount of GPUs available, if desired by user
        total_gpus = torch.cuda.device_count()
        if cfg.cuda.num_gpus < 0:
            OmegaConf.update(cfg, "cuda.num_gpus", total_gpus)

        # user may have requested too many GPUs
        log = f"Requested {cfg.cuda.num_gpus} of {total_gpus} available GPUs"
        if cfg.cuda.num_gpus > total_gpus:
            s = "s" if total_gpus > 1 else ""
            logger.warning(log)
            logger.warning(f"Resuming with {total_gpus} GPU{s}")
            OmegaConf.update(cfg, "cuda.num_gpus", total_gpus)
        else:
            logger.info(log)

        # exclude GPUs that have not been made visible
        if len(cfg.cuda.visible_devices) < cfg.cuda.num_gpus:
            num_visible_gpus = len(cfg.cuda.visible_devices)
            s = "s" if num_visible_gpus > 1 else ""
            logger.warning(f"Only {num_visible_gpus} visible device{s}")
            OmegaConf.update(cfg, "cuda.num_gpus", num_visible_gpus)

    if cfg.cuda.num_gpus == 0:
        logger.error("Trainer assumes GPU-based training")
        logger.error("To use CPU, remove DDP and mixed precision")
        exit(1)

    devices = range(len(cfg.cuda.visible_devices))[: cfg.cuda.num_gpus]
    cuda = ", ".join([f"cuda:{cfg.cuda.visible_devices[i]}" for i in devices])
    logger.info(f"Using {cuda if cfg.cuda.num_gpus > 0 else 'cpu'}")

    return cfg


def validate_config(cfg: DictConfig) -> DictConfig:
    """
    Sanity check the config with type hints, and dynamically configure
    settings if necessary (ex. GPU availability).

    Error message will indicate whether it was a type mismatch, missing
    mandatory field, etc.
    """
    OmegaConf.register_new_resolver("pop", _pop)
    cfg = OmegaConf.merge(OmegaConf.structured(Schema), cfg)
    OmegaConf.resolve(cfg)

    # make sure path/to/repo exists for relative interpolated paths to work
    if not cfg.repo_dir.exists():
        logging.error(f"Could not locate repo at: {cfg.repo_dir}")
        exit(1)

    try:
        if cfg.experiment_dir.exists() and cfg.overwrite:
            logger.warning("Overwriting experiment directory")
        cfg.experiment_dir.mkdir(exist_ok=cfg.overwrite, parents=True)
    except FileExistsError:
        logging.error(f"Experiment already exists at {cfg.experiment_dir}")
        exit(1)

    # # set up file handler with log file path from config
    # # will need to be reset in "a" mode after mp.spawn()
    # file_handler = logging.FileHandler(filename=cfg.log, mode="w")
    # file_handler.setFormatter(logging.root.handlers[-1].formatter)
    # logging.getLogger().addHandler(file_handler)

    # ensure GPU(s) is/are available
    cfg = configure_device(cfg)

    if cfg.seed is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # write the final interpolated config file to experiment directory
    cfg_dict = _convert_path_to_str(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg_dict, f=cfg.experiment_dir / "config.yaml", resolve=True)

    return cfg
