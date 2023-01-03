#!/usr/bin/env python
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.arch.lstm import LSTM
from trainer.config import parse_config, process_config
from trainer.data.audioloader import get_dl
from trainer.distributed import ddp, spawn
from trainer.evaluate import Evaluator
from trainer.logging import create_file_handler, create_rich_handler
from trainer.train import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
log_config = OmegaConf.to_object(OmegaConf.load(PROJECT_ROOT / "log.yml"))
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)


def read_config() -> DictConfig:

    if len(sys.argv) == 2:  # load configuration file from command line
        if (yaml_path := Path(sys.argv[1])).suffix not in {".yml", ".yaml"}:
            logger.error("Only YAML configuration files are supported")
            exit(1)
        yaml = OmegaConf.load(yaml_path)

    else:  # compose with Hydra
        with initialize("../config", None):
            yaml = compose("example_motion")

    # parse and interpolate yaml dict
    cfg = parse_config(yaml)

    # root logging handlers are instantiated here instead of log.yml so default
    # Console object used by Rich is shared between RichHandler and RichProgress
    # https://github.com/Textualize/rich/issues/1379
    logger.root.addHandler(create_rich_handler())
    logger.root.addHandler(create_file_handler(cfg.log))

    # validate config file after logging has been fully set up
    process_config(cfg)

    return cfg


def initialize_lstm(
    cfg: DictConfig,
    dl: torch.utils.data.DataLoader,
    rank: int,
) -> torch.nn.Module:
    """
    Create LSTM with hyperparameters from config and input/output size from DL.
    """
    model = LSTM(
        input_size=dl.dataset[0][0].shape[-1],
        out_features=dl.dataset[0][1].data.shape[-1],
        **cfg.arch.lstm,  # unpack the hyperparams as kwargs
    ).to(rank)

    # clone model across all GPUs
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[rank])

    return model


def evaluate(cfg: DictConfig) -> None:

    t_dl = get_dl(cfg.test.data.x_dir, cfg.test.data.y_dir)
    model = initialize_lstm(cfg, t_dl, cfg.cuda.visible_devices[0])
    evaluator = Evaluator(cfg, model, t_dl, checkpoint="checkpoint_best.pth")
    logger.info("Evaluating")
    evaluator.evaluate()


def train(cfg: DictConfig, queue: mp.Queue, **kwargs) -> None:
    """
    Load data and train a model. When DDP is enabled, this function will be
    executed across N subprocesses corresponding to the number of GPUs.
    """

    # Set seed at the beginning of the training process. It is crucial to set
    # it here with DDP so that all subprocesses initialize the model with the
    # same weights.
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = torch.cuda.current_device()

    # create DataLoaders
    t_dl, v_dl = get_dl(
        cfg.train.data.x_dir,
        cfg.train.data.y_dir,
        batch_size=cfg.train.batch_size,
        valid=True,
    )

    # instantiate model
    model = initialize_lstm(cfg, t_dl, rank)

    # instantiate optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        cfg.train.lr,
        betas=(0.9, 0.99),
        eps=1e-05,
    )

    # define regression metrics to track; loss must be first
    metrics = [torch.nn.MSELoss(), torch.nn.L1Loss()]

    # collect model components into a Trainer object
    args = [cfg, model, optimizer, t_dl, v_dl, metrics, queue, rank]
    trainer = Trainer(*args)
    logger.info("Training")
    trainer.train()


def main() -> None:

    cfg = read_config()

    # torch requires this before any multiprocess object creation
    mp.set_start_method("spawn")

    # to share memory between processes, pass messages via queue
    queue, log_queue = mp.Queue(), mp.Queue()
    listener = logging.handlers.QueueListener(log_queue, *logger.root.handlers)
    listener.start()

    # train with DistributedDataParallel if more than one GPU is requested
    # otherwise delegate training to a subprocess so main can handle stdout
    args = [train, cfg, queue, log_queue]
    ddp(*args) if cfg.cuda.num_gpus > 1 else spawn(*args)

    # run evaluation with 1 GPU from main process only (for now?)
    evaluate(cfg)

    # log_queue.put_nowait(None)
    listener.stop()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception):
        logger.exception("")
        exit(1)
