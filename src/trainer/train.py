"""
A PyTorch training loop template with built-in support for
DistributedDataParallel and automatic mixed precision.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.distributed.algorithms.join import Join
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .metrics import AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
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

        logging.debug("Setting up Trainer")

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

    def _compare(self, epoch_loss, best_loss):
        """
        Determine whether validation loss has improved for current epoch. For
        metrics that increase (i.e. accuracy), set Trainer comparison="gt"
        """
        if self.comparison == "lt":
            compare = np.less
            self.best_loss = min(epoch_loss, best_loss)
        elif self.comparison == "gt":
            compare = np.greater
            self.best_loss = max(epoch_loss, best_loss)
        else:
            raise NotImplementedError(f'`comparison` must be "lt" or "gt"')
        return compare(epoch_loss, best_loss)

    def _run_batches(self, dl: DataLoader, train: bool) -> None:

        for x, y in dl:

            x, y = x.to(self.rank), y.to(self.rank)

            # automatic mixed precision (amp)
            with torch.cuda.amp.autocast():

                # forward pass
                if not train and torch.distributed.is_initialized():
                    # eval occurs on rank 0 only: get non-replicated model
                    y_hat = self.model.module(x)
                else:
                    y_hat = self.model(x)

                # the first metric in the list is the loss function
                batch_metrics = [metric(y_hat, y) for metric in self.metrics]

                if train:
                    self.scaler.scale(batch_metrics[0]).backward()  # amp
                    self.scaler.step(self.optimizer)  # amp
                    self.scaler.update()  # amp
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                for i, m in enumerate(self.metric_averages):
                    m.update(batch_metrics[i])

            # pbar.comment = ""

    def _predict(self, *args, **kwargs) -> None:
        # use context manager necessary for variable-length batches with DDP
        # Source: https://pytorch.org/tutorials/advanced/generic_join.html
        if torch.distributed.is_initialized():
            with Join([self.model]):
                self._run_batches(*args, **kwargs)
        else:
            self._run_batches(*args, **kwargs)

    def predict(self, dl: DataLoader, train: bool = False) -> list[float]:

        """
        In distributed mode, calling the set_epoch() method at the beginning
        of each epoch before creating the DataLoader iterator is necessary to
        make shuffling work properly across multiple epochs. Otherwise, the
        same ordering will be always used. Source:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        """
        # if torch.distributed.is_initialized():
        #     dl.sampler.set_epoch(self.current_epoch)

        # switch off grad engine if applicable
        self.model.train() if train else self.model.eval()

        # update metric averages
        self._predict(dl, train=train)

        # wait for all GPU processes to finish
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # valid metrics do not exist on rank >= 1, default to 0.0
        return [
            metric.avg.item() if isinstance(metric.avg, torch.Tensor) else 0.0
            for metric in self.metric_averages
        ]

    def train(self, checkpoint: str | None = None) -> None:

        if checkpoint:
            self.load_checkpoint(self.cfg.experiment_dir / checkpoint)
            logging.info(f"Resuming from epoch {self.start_epoch}")

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

        logger.debug(f"Loading {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # load variables from checkpoint
        self.start_epoch = checkpoint["epoch"]
        self.metrics = checkpoint["metrics"]
        if torch.distributed.is_initialized():
            self.model.module.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])

    def close(self) -> None:
        """
        Close any objects that need it.
        """
        for obj in self.closable:
            obj.close()
