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

import numpy as np
import torch
from fastprogress import progress_bar
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.distributed.algorithms.join import Join
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .metrics import AverageMeter
from .utils import strip_module_prefix

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

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.t_dl = t_dl
        self.v_dl = v_dl
        self.metrics = metrics
        self.rank = rank
        self.comparison = comparison

        # path for checkpoints
        experiment_dir = self.cfg.checkpoint_dir / self.cfg.experiment
        self.checkpoint_dir = experiment_dir / dir(self.cfg.arch)[0]

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

        # list of objects that need to be closed at the end of training
        self.closable = []

    def _compare(self, epoch_loss, best_loss):
        if self.comparison == "lt":
            compare = np.less
            self.best_loss = min(epoch_loss, best_loss)
        elif self.comparison == "gt":
            compare = np.greater
            self.best_loss = max(epoch_loss, best_loss)
        else:
            raise NotImplementedError(f'Trainer `comp` must be "lt" or "gt"')
        return compare(epoch_loss, best_loss)

    def _run_batches(
        self,
        dl: DataLoader,
        meters: list[AverageMeter],
        train: bool,
    ) -> None:

        for x, y in dl:

            x, y = x.to(self.rank), y.to(self.rank)

            # automatic mixed precision (amp)
            with torch.cuda.amp.autocast():

                # forward pass
                # eval occurs on rank 0 only; model.module gets non-replicated model
                if not train and torch.distributed.is_initialized():
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

                for i, m in enumerate(meters):
                    m.update(batch_metrics[i])

            # pbar.comment = ""

    def _predict(self, *args) -> None:
        # use context manager necessary for variable-length batches with DDP
        # Source: https://pytorch.org/tutorials/advanced/generic_join.html
        if torch.distributed.is_initialized():
            with Join([self.model]):
                self._run_batches(*args)
        else:
            self._run_batches(*args)

    def predict(
        self,
        dl: DataLoader,
        train: bool = False,
        test: bool = False,
    ) -> list[float] | list[torch.Tensor]:

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

        # wrap DataLoader iterable in a progress bar
        # pbar = progress_bar(dl, leave=False)

        if test:
            return [self.model(x.to(self.cfg.device)).detach() for x, _ in dl]

        # collect metric averages
        average_meters = [AverageMeter() for _ in self.metrics]

        self._predict(dl, average_meters, train)

        # wait for all GPU processes to finish
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # valid metrics do not exist on rank >= 1, default to 0.0
        return [
            metric.val.item() if isinstance(metric.val, torch.Tensor) else 0.0
            for metric in average_meters
        ]

    def train(self, resume: bool = False) -> None:

        if resume:
            self.load_checkpoint(best=False)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            train_metrics = self.predict(self.t_dl, train=True)
            valid_metrics = self.predict(self.v_dl)

            # determine whether model performance has improved in this epoch
            is_best = self._compare(valid_metrics[0], self.best_loss)

            # save model parameters and metadata
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,  # add 1 to set start_epoch if resuming
                    "metrics": self.metrics,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.lr_scheduler.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                },
                is_best=is_best,
            )

        logger.info(f"Training completed")
        self.close()

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        name: str = "checkpoint",
        checkpoint_dir: Path | None = None,
        is_best: bool = False,
    ) -> None:
        """
        Serialize a PyTorch model with its associated parameters as a
        checkpoint dict object.
        """

        # only save the checkpoint from the main process
        if self.rank != 0:
            return

        # keys expected by load_checkpoint()
        REQUIRED_KEYS = {
            "epoch",
            "metrics",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
        }
        assert (
            checkpoint.keys() >= REQUIRED_KEYS
        ), f"Checkpoint dict must contain {REQUIRED_KEYS}"

        # determine the path to save the checkpoint
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        torch.save(checkpoint, checkpoint_dir / f"{name}.pth")

        if is_best:
            torch.save(checkpoint, checkpoint_dir / f"{name}_best.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> None:

        logger.info(f"Loading {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # models trained with DDP are incompatible with non-DDP models
        if not torch.distributed.is_initialized():
            strip_module_prefix(checkpoint)

        # load variables from checkpoint
        self.start_epoch = checkpoint["epoch"]
        self.metrics = checkpoint["metrics"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])

        logger.info(f"Loaded checkpoint @ epoch {checkpoint['epoch']}")

    def close(self) -> None:
        """
        Close any objects that need it.
        """
        for obj in self.closable:
            obj.close()
