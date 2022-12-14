#!/usr/bin/env python

"""
https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from MLtools.utils.logging import configure_logger
from trainer.config import validate_config
from trainer.data import get_dl
from trainer.models import LSTM
from trainer.train import Trainer

logger = logging.getLogger(__name__)
configure_logger()
logging.getLogger("torch").setLevel(logging.WARNING)  # ignore DDP info


def parse_config() -> DictConfig:

    if len(sys.argv) == 2:  # load configuration file from command line
        if (yaml_path := Path(sys.argv[1])).suffix not in {".yml", ".yaml"}:
            logger.error("Only YAML configuration files are supported")
            exit(1)
        yaml = OmegaConf.load(yaml_path)

    else:  # compose with Hydra
        with initialize("../config", None):
            yaml = compose("example_motion_ddp")

    return validate_config(yaml)


def ddp_setup(rank: int, world_size: int) -> None:
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # TODO: only log from the main process


def main(rank: int, cfg: DictConfig) -> None:

    # configure current worker within the DistributedDataParallel context
    if cfg.cuda.num_gpus > 1:
        ddp_setup(rank, cfg.cuda.num_gpus)

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
    if torch.distributed.is_initialized():
        model = DDP(model.to(rank), device_ids=[rank])
    else:
        model = model.to(rank)

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
    trainer = Trainer(cfg, model, optimizer, t_dl, v_dl, metrics, rank)

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

    # terminate DDP worker
    if torch.distributed.is_initialized():
        destroy_process_group()


if __name__ == "__main__":

    cfg = parse_config()

    try:
        if cfg.cuda.num_gpus <= 1:
            main(rank=cfg.cuda.visible_devices[0], cfg=cfg)
        else:
            mp.spawn(main, args=([cfg]), nprocs=cfg.cuda.num_gpus)

    except (KeyboardInterrupt, Exception):
        logger.exception("")
        exit(1)
