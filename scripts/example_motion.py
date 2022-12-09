#!/usr/bin/env python

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from MLtools.utils.logging import configure_logger
from trainer.config import type_validate_config
from trainer.data import get_dl
from trainer.models import LSTM
from trainer.train import Trainer

logger = logging.getLogger(__name__)
configure_logger()


def parse_config() -> DictConfig:

    if len(sys.argv) == 2:  # load configuration file from command line
        if (yaml_path := Path(sys.argv[1])).suffix not in {".yml", ".yaml"}:
            logger.error("Only YAML configuration files are supported")
            exit(1)
        yaml = OmegaConf.load(yaml_path)

    else:  # compose with Hydra
        with initialize("../config", None):
            yaml = compose("example_motion")

    return type_validate_config(yaml)


def main(cfg: DictConfig) -> None:

    # create PyTorch DataLoaders
    t_dl, v_dl = get_dl(
        cfg.train.data.x_dir,
        cfg.train.data.y_dir,
        batch_size=cfg.train.batch_size,
        valid=True,
    )

    # instantiate PyTorch model
    model = LSTM(
        input_size=t_dl.dataset[0][0].shape[-1],
        out_features=t_dl.dataset[0][1].data.shape[-1],
        **cfg.arch.lstm,  # unpack the rest of the hyperparams as kwargs
    )
    model.to(cfg.device)

    # instantiate optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        cfg.train.lr,
        betas=(0.9, 0.99),
        eps=1e-05,
    )

    # define regression metrics to track; loss must be first
    metrics = [torch.nn.MSELoss(), torch.nn.L1Loss()]

    # collect model components into a Trainer objects
    trainer = Trainer(cfg, model, optimizer, t_dl, v_dl, metrics)

    logger.info("Training")
    trainer.train()

    # create PyTorch DataLoader for inference
    t_dl = get_dl(
        cfg.test.data.x_dir,
        cfg.test.data.y_dir,
        batch_size=cfg.test.batch_size,
    )

    logger.info("Running model in inference mode")
    logger.info(f"Predicting {len(t_dl)} samples")
    trainer.load_checkpoint(trainer.checkpoint_dir / "checkpoint_best.pth")
    predictions = trainer.predict(t_dl, test=True)
    logger.info(f"{len(predictions)} predictions collected")


if __name__ == "__main__":

    cfg = parse_config()

    try:
        main(cfg)

    except (KeyboardInterrupt, Exception):
        logger.exception("")
        exit(1)
