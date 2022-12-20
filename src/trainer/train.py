"""
A PyTorch training loop template with built-in support for
DistributedDataParallel and automatic mixed precision.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from fastprogress import progress_bar
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .base import Base
from .metrics import AverageMeter

logger = logging.getLogger(__name__)


class Trainer(Base):
    """
    An object for handling training. Create a new one for each training run.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        optimizer: torch.nn.Module,
        t_dl: DataLoader,
        v_dl: DataLoader,
        metrics: list[Callable],
        rank: int = 0,
        comparison: str = "lt",  # {"lt", "gt"}
    ) -> None:

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.t_dl = t_dl
        self.v_dl = v_dl
        self.metrics = metrics
        self.rank = rank
        self.comparison = comparison

        # training
        self.start_epoch = self.current_epoch = 0
        self.best_loss = torch.inf if self.comparison == "lt" else -torch.inf
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            self.cfg.train.lr,
            epochs=self.cfg.train.epochs,
            steps_per_epoch=len(self.t_dl),
        )
        self.scaler = GradScaler()  # automatic mixed precision
        self.metric_averages = [AverageMeter() for _ in self.metrics]

        # list of objects that need to be closed at the end of training
        self.closable = []

    def train(self, resume: bool = False) -> None:

        if resume:
            self.load_checkpoint(best=False)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            self.current_epoch = epoch

            _ = self.predict(self.t_dl, train=True)
            val_loss, _ = self.predict(self.v_dl)

            # save model parameters and metadata
            self.save_checkpoint(val_loss)

        logger.debug(f"Training completed")
        self.close()

    def save_checkpoint(
        self,
        val_loss: float,
        name: str = "checkpoint",
    ) -> None:
        """
        Serialize a PyTorch model with its associated parameters as a
        checkpoint dict object.
        """

        if self.rank != 0:  # only save the checkpoint from the main process
            return

        # distributed models are wrapped in a DDP object
        if torch.distributed.is_initialized():
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            "epoch": self.current_epoch + 1,  # add 1 for start_epoch if resume
            "metrics": self.metrics,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.cfg.experiment_dir / f"{name}.pth")

        # determine whether model performance has improved in this epoch
        if self._compare(val_loss, self.best_loss):
            torch.save(checkpoint, self.cfg.experiment_dir / f"{name}_best.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        super().load_checkpoint(checkpoint_path)
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(self.checkpoint["scheduler_state"])
        self.scaler.load_state_dict(self.checkpoint["scaler_state"])

    def close(self) -> None:
        """
        Close any objects that need it.
        """
        for obj in self.closable:
            obj.close()
