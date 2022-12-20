"""
A PyTorch base class for operations shared between training and evaluation.
Built-in support for DistributedDataParallel and automatic mixed precision.

Created: Dec 2022 by Eric DeMattos
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.distributed.algorithms.join import Join
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Base:
    def __init__(self) -> None:
        pass

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

    def load_checkpoint(self, checkpoint_path: Path) -> None:

        logger.debug(f"Loading {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path)

        # load variables from checkpoint
        self.start_epoch = self.checkpoint["epoch"]
        self.metrics = self.checkpoint["metrics"]
        if torch.distributed.is_initialized():
            self.model.module.load_state_dict(self.checkpoint["model_state"])
        else:
            self.model.load_state_dict(self.checkpoint["model_state"])
