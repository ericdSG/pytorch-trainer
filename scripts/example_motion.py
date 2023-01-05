#!/usr/bin/env python
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from trainer.arch import create_model
from trainer.config import parse_config
from trainer.data import create_dl
from trainer.distributed import spawn
from trainer.evaluate import Evaluator
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
    return parse_config(yaml)


def evaluate(cfg: DictConfig, rank: int = torch.cuda.current_device()) -> None:

    t_dl = create_dl(cfg.test.data.x_dir, cfg.test.data.y_dir)
    model = create_model(cfg, t_dl, rank)
    evaluator = Evaluator(cfg, model, t_dl, checkpoint="checkpoint_best.pth")
    logger.info("Evaluating")
    evaluator.evaluate()


def train(
    cfg: DictConfig,
    queue: torch.multiprocessing.Queue,
    rank: int = torch.cuda.current_device(),
    **kwargs,
) -> None:
    """
    Load data and train a model.
    """

    # Set seed at the beginning of the training process. It is crucial to set
    # it here when using DDP so that all subprocesses initialize the model with
    # the same weights.
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # create DataLoaders
    t_dl, v_dl = create_dl(
        cfg.train.data.x_dir,
        cfg.train.data.y_dir,
        batch_size=cfg.train.batch_size,
        valid=True,
    )

    # instantiate model
    model = create_model(cfg, t_dl, rank)

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
    trainer = Trainer(cfg, model, optimizer, t_dl, v_dl, metrics, queue, rank)
    logger.info("Training")
    trainer.train()


def main() -> None:

    cfg = read_config()

    # torch requires this before any multiprocess object creation
    torch.multiprocessing.set_start_method("spawn")

    # logs from all process are handled by QueueListener in main process
    log_queue = torch.multiprocessing.Queue()
    listener = logging.handlers.QueueListener(log_queue, *logger.root.handlers)
    listener.start()

    # delegate training to subprocess(es)
    spawn(train, cfg, log_queue)

    # run evaluation with 1 GPU from main process only (for now)
    OmegaConf.update(cfg, "cuda.num_gpus", 1)
    OmegaConf.update(cfg, "cuda.ddp", False)
    evaluate(cfg)

    # log_queue.put_nowait(None)
    listener.stop()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception):
        logger.exception("")
        exit(1)
