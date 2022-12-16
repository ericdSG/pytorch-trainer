"""
A PyTorch training loop template with built-in support for
DistributedDataParallel and automatic mixed precision.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Tuple

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

REQUIRED_CHECKPOINT_KEYS = [
    "epoch",
    "metrics",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
]


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

    def predict(
        self,
        dl: DataLoader,
        train: bool = False,
        test: bool = False,
    ) -> list[float] | list[Tuple[torch.Tensor, str]]:

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
            preds = [self.model(x.to(self.rank)).detach() for x, _ in dl]
            utt_ids = [Path(path).stem for path in dl.dataset.labels]
            return [(pred, utt_id) for pred, utt_id in zip(preds, utt_ids)]

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

    def train(self, resume: bool = False) -> None:

        if resume:
            self.load_checkpoint(best=False)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            self.current_epoch = epoch

            train_metrics = self.predict(self.t_dl, train=True)
            valid_metrics = self.predict(self.v_dl)

            # determine whether model performance has improved in this epoch
            is_best = self._compare(valid_metrics[0], self.best_loss)

            # save model parameters and metadata
            self.save_checkpoint(is_best=is_best)

        logger.debug(f"Training completed")
        self.close()

    def save_checkpoint(
        self,
        name: str = "checkpoint",
        is_best: bool = False,
    ) -> None:
        """
        Serialize a PyTorch model with its associated parameters as a
        checkpoint dict object.
        """

        if self.rank != 0:  # only save the checkpoint from the main process
            return

        checkpoint = {
            "epoch": self.current_epoch + 1,  # add 1 for start_epoch if resume
            "metrics": self.metrics,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }

        missing_keys = sorted(REQUIRED_CHECKPOINT_KEYS - checkpoint.keys())
        assert not missing_keys, f"{missing_keys} must be saved to checkpoint"

        torch.save(checkpoint, self.cfg.experiment_dir / f"{name}.pth")
        if is_best:
            torch.save(checkpoint, self.cfg.experiment_dir / f"{name}_best.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> None:

        # only need to log the checkpoint path relative to repository root
        abbrev_checkpoint_path = "/".join(str(checkpoint_path).split("/")[-4:])
        logger.debug(f"Loading {abbrev_checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # models trained with DDP are incompatible with non-DDP models
        if not torch.distributed.is_initialized():
            strip_module_prefix(checkpoint)

        # load variables from checkpoint
        keys = REQUIRED_CHECKPOINT_KEYS
        self.start_epoch = checkpoint[keys.pop(0)]
        self.metrics = checkpoint[keys.pop(0)]
        self.model.load_state_dict(checkpoint[keys.pop(0)])
        self.optimizer.load_state_dict(checkpoint[keys.pop(0)])
        self.lr_scheduler.load_state_dict(checkpoint[keys.pop(0)])
        self.scaler.load_state_dict(checkpoint[keys.pop(0)])
        assert not keys, f"{keys} not loaded from checkpoint"

        logger.debug(f"Loaded checkpoint @ epoch {checkpoint['epoch']}")

    def close(self) -> None:
        """
        Close any objects that need it.
        """
        for obj in self.closable:
            obj.close()
