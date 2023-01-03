"""
A PyTorch training loop template with built-in support for
DistributedDataParallel and automatic mixed precision.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

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
        queue: torch.multiprocessing.Queue,
        rank: int,
        comparison: Literal["lt", "gt"] = "lt",
    ) -> None:

        logger.debug("Setting up Trainer")

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.t_dl = t_dl
        self.v_dl = v_dl
        self.metrics = metrics
        self.rank = rank
        self.comparison = comparison
        self.queue = queue

        # training
        self.start_epoch = self.epoch = 0
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

    def _compare(self, epoch_loss: float, best_loss: float) -> bool:
        """
        Determine whether validation loss has improved for current epoch. For
        metrics that increase, e.g. accuracy, set Trainer(comparison="gt").
        """
        losses = (epoch_loss, best_loss)
        if self.comparison == "lt":
            self.best_loss = min(*losses)
            return np.less(*losses)
        elif self.comparison == "gt":
            self.best_loss = max(*losses)
            return np.greater(*losses)
        else:
            raise NotImplementedError(f'`comparison` must be "lt" or "gt"')

    def _run_batch(self, x, y, train) -> list[torch.Tensor]:

        x, y = x.to(self.rank, non_blocking=True), y.to(
            self.rank, non_blocking=True
        )

        # forward pass
        if not train and torch.distributed.is_initialized():
            # get non-replicated model for eval
            # https://discuss.pytorch.org/t/99867/11
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

        return batch_metrics

    def _run_batch_amp(self, *args, **kwargs) -> list[torch.Tensor]:
        with torch.cuda.amp.autocast():
            return self._run_batch(*args, **kwargs)

    def _predict(self, dl: DataLoader, train: bool) -> list[AverageMeter]:

        # switch off grad engine if applicable
        self.model.train() if train else self.model.eval()

        # collect metric values for current epoch
        meters = [AverageMeter(m, self.rank) for m in self.metrics]

        # keep track of how many samples have been processed for progress bars
        count = 0

        for x, y in dl:

            metrics = self._run_batch_amp(x, y, train)

            # keep losses as tensors, but remove computational graph
            for i, m in enumerate(meters):
                m.update(metrics[i].detach())

            # propagate information to main process for dashboard
            count += x.shape[0]
            self.queue.put((self.rank, count, len(dl), self.epoch + 1))

        # need to update dashboardwhen current rank has no samples to process
        if len(dl) == 0:
            self.queue.put((self.rank, 0, 0, self.epoch + 1))

        return meters

    def _predict_ddp(self, dl: DataLoader, train: bool) -> list[AverageMeter]:

        # update DistributedGroupSampler epoch for deterministic shuffling
        dl.batch_sampler.set_epoch(self.epoch)

        # use context manager necessary for variable-length batches with DDP
        # Source: https://pytorch.org/tutorials/advanced/generic_join.html
        with Join([self.model]):
            meters = self._predict(dl, train=train)

        # wait for all GPU processes to finish
        torch.distributed.barrier()

        # when training with DDP, model gradients are averaged and automatically
        # applied across all processes after the backward pass. OTOH, metrics
        # are local to each process: they only reflect the subset of the data
        # allocated to each GPU. to sync, use distributed methods for NCCL
        # backend, which updates the variable in-place on all processes
        # https://pytorch.org/docs/stable/distributed.html
        # https://discuss.pytorch.org/t/93306/4

        for meter in meters:
            # must be cuda tensors (without computation graph)
            torch.distributed.all_reduce(meter.avg)  # TODO: reduce strategy
            torch.distributed.all_reduce(meter.count)  # TODO: reduce strategy
            torch.distributed.all_reduce(meter.sum)  # TODO: reduce strategy
            torch.distributed.all_reduce(meter.val)  # TODO: reduce strategy

        return meters

    def predict(self, dl: DataLoader, train: bool) -> list[float]:

        if torch.distributed.is_initialized():
            meters = self._predict_ddp(dl, train=train)
        else:
            meters = self._predict(dl, train=train)

        return [meter.avg.item() for meter in meters]

    def train(self, checkpoint: str | None = None) -> None:

        if checkpoint:
            self.load_checkpoint(self.cfg.experiment_dir / checkpoint)
            logger.info(f"Resuming from epoch {self.start_epoch}")

        # # log header of tracked metrics TENTATIVE
        # logger.info(
        #     f"Monitoring {self.metrics[0].__class__.__name__} on valid set (*)"
        # )
        # float_precision = 8
        # str_width = 10
        # log = "Epoch | *"
        # for metric in self.metrics:
        #     metric_name = metric.__class__.__name__
        #     for split in ["t", "v"]:
        #         metric = f"{split}_{metric_name}"
        #         log += f" | {metric[:str_width]:^{str_width}}"
        # # log += f" | {'time':^5}"
        # logger.info(log)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            self.epoch = epoch

            train_metrics = self.predict(self.t_dl, train=True)
            valid_metrics = self.predict(self.v_dl, train=False)

            # save model parameters and metadata
            is_best = self._compare(valid_metrics[0], self.best_loss)
            self.save_checkpoint(is_best)

            # # log metrics (fixed width capped at float_precision) TENTATIVE
            # log = f"{self.epoch:>{len('epoch')}} | "
            # log += "*" if is_best else " "
            # for train_value, valid_value in zip(train_metrics, valid_metrics):
            #     train_value = f"{train_value:0.{float_precision}f}"
            #     valid_value = f"{valid_value:0.{float_precision}f}"
            #     log += f" | {train_value[:str_width]} | {valid_value[:str_width]}"
            # # log += f" | {epoch_time}"
            # # log += f" | {loss}"
            # logger.info(log)

        logger.debug(f"Training completed")
        self.close()

    def save_checkpoint(
        self,
        is_best: bool,
        name: str = "checkpoint",
    ) -> None:
        """
        Serialize a PyTorch model with its associated parameters as a
        checkpoint dict object.
        """

        if self.rank != 0:  # only save the checkpoint from the main subprocess
            return

        # distributed models are wrapped in a DDP object
        if torch.distributed.is_initialized():
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            "epoch": self.epoch + 1,  # add 1 for start_epoch if resume
            "metrics": self.metrics,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.cfg.experiment_dir / f"{name}.pth")

        # determine whether model performance has improved in this epoch
        if is_best:
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
