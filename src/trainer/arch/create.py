from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .lstm import LSTM

logger = logging.getLogger(__name__)


def _get_model(cfg: DictConfig, dl: DataLoader, **kwargs) -> torch.nn.Module:
    """
    Get model name and hyperparameters from config and
    input/output size based on shape of transformed features.
    """

    arch = list(cfg.arch.keys()).pop(0)
    x_dim = dl.dataset[0][0].shape[-1]
    y_dim = dl.dataset[0][1].data.shape[-1]

    if arch == "lstm":
        return LSTM(input_size=x_dim, out_features=y_dim, **cfg.arch.lstm)
    else:
        raise NotImplementedError()


def create_model(
    cfg: DictConfig,
    dl: DataLoader,
    rank: int,
) -> torch.nn.Module | DP | DDP:

    model = _get_model(**locals()).to(rank)

    if cfg.cuda.ddp:
        return DDP(model, device_ids=[rank])
    elif cfg.cuda.num_gpus > 1:
        return DP(model, device_ids=[i for i in range(cfg.cuda.num_gpus)])
    else:
        return model
