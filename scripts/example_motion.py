#!/usr/bin/env python
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.arch.lstm import LSTM
from trainer.config import validate_config
from trainer.data.audioloader import get_dl
from trainer.distributed import train_ddp
from trainer.evaluate import Evaluator
from trainer.train import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
log_config = OmegaConf.to_object(OmegaConf.load(PROJECT_ROOT / "log.yml"))
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)


def parse_config() -> DictConfig:

    if len(sys.argv) == 2:  # load configuration file from command line
        if (yaml_path := Path(sys.argv[1])).suffix not in {".yml", ".yaml"}:
            logger.error("Only YAML configuration files are supported")
            exit(1)
        yaml = OmegaConf.load(yaml_path)

    else:  # compose with Hydra
        with initialize("../config", None):
            yaml = compose("example_motion")

    return validate_config(yaml)


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


def train(cfg: DictConfig, queue: mp.Queue | None = None, **kwargs) -> None:

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = cfg.cuda.visible_devices[0]

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
    trainer = Trainer(
        cfg, model, optimizer, t_dl, v_dl, metrics, rank, queue=queue
    )
    logger.info("Training")
    trainer.train()


def main() -> None:

    cfg = parse_config()

    # set up DistributedDataParallel if more than one GPU requested
    train_ddp(cfg, train) if cfg.cuda.num_gpus > 1 else train(cfg)

    # run evaluation with 1 GPU from main process only (for now?)
    evaluate(cfg)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception):
        logger.exception("")
        exit(1)
