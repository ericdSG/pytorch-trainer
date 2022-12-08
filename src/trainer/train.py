"""
A PyTorch training loop template, implemented to mimic the functionality of
the previously used FastAI training loop, including some of its defaults,
callbacks, and metrics.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Union

import torch
from fastprogress import progress_bar
from omegaconf import DictConfig
from torch.optim import AdamW
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
        t_dl: DataLoader,
        v_dl: DataLoader,
        metrics: list[Callable],
        optimizer: torch.nn.Module | None = None,
    ) -> None:

        self.cfg = cfg

        # data
        self.t_dl = t_dl
        self.v_dl = v_dl

        # model
        self.model = model.to(self.cfg.device)

        # path for checkpoints
        experiment_dir = self.cfg.checkpoint_dir / self.cfg.experiment
        self.checkpoint_dir = (
            experiment_dir / self.model.__class__.__name__.lower()
        )

        # define optimizer
        if not optimizer:
            self.optimizer = AdamW(
                self.model.parameters(),
                self.cfg.train.lr,
                betas=(0.9, 0.99),
                eps=1e-05,
            )
        else:
            self.optimizer = optimizer

        # training
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.optimizer = self.optimizer
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            self.cfg.train.lr,
            epochs=self.cfg.train.epochs,
            steps_per_epoch=len(self.t_dl),
        )

        # evaluation
        self.metrics = metrics
        logger.info(f"Monitoring {self.metrics[0].__class__.__name__}")

        # list of objects that need to be closed at the end of training
        self.closable = []

    def train(self, resume: bool = False) -> None:

        if resume:
            self.load_checkpoint(best=False)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            train_loss, train_acc = self.predict(self.t_dl, train=True)
            valid_loss, valid_acc = self.predict(self.v_dl)

            # determine whether model performance has improved in this epoch
            is_best = valid_acc > self.best_val_acc
            self.best_val_acc = max(valid_acc, self.best_val_acc)

            # save model parameters and metadata
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,  # add 1 to set start_epoch if resuming
                    "metrics": self.metrics,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.lr_scheduler.state_dict(),
                },
                is_best=is_best,
            )

        logger.info(f"Training completed")
        self.close()

    def predict(
        self,
        dl: DataLoader,
        train: bool = False,
        predictions: bool = False,
    ) -> Union[list[float], list[torch.Tensor]]:

        # switch off grad engine if applicable
        self.model.train() if train else self.model.eval()

        pbar = progress_bar(dl, leave=False)

        if predictions:
            return [self.model(x.to(self.cfg.device)).detach() for x, _ in pbar]

        # collect metric averages
        average_meters = [AverageMeter() for _ in self.metrics]

        for x, y in pbar:

            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            y_hat = self.model(x)

            # the first metric in the list is the loss function
            batch_metrics = [metric(y_hat, y) for metric in self.metrics]

            if train:
                batch_metrics[0].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            for i, m in enumerate(average_meters):
                m.update(batch_metrics[i])

            pbar.comment = ""

        return [metric.val for metric in average_meters]

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

        # keys expected by load_checkpoint()
        REQUIRED_KEYS = {
            "epoch",
            "metrics",
            "model_state",
            "optimizer_state",
            "scheduler_state",
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

        # load variables from checkpoint
        self.start_epoch = checkpoint["epoch"]
        self.metrics = checkpoint["metrics"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])

        logger.info(f"Loaded checkpoint @ epoch {checkpoint['epoch']}")

    def close(self) -> None:
        """
        Close any objects that need it.
        """
        for obj in self.closable:
            obj.close()


if __name__ == "__main__":
    raise NotImplementedError("train.py should not be run directly.")
