from __future__ import annotations

import logging
from typing import Tuple

import torch.multiprocessing as mp
from omegaconf import DictConfig
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table

logger = logging.getLogger(__name__)


class EpochMonitor:
    def __init__(
        self,
        num_ranks: int,
        train: bool = True,
        valid: bool = False,
    ) -> None:

        assert not (train and valid) or (not train and not valid)

        self.num_ranks = num_ranks
        self.train = train
        self.valid = valid

        self._reset()

    def _get_task(self) -> int:
        return 0 if self.train else 1

    def _reset(self) -> None:
        self.task = self._get_task()
        self.completed = [False for _ in range(self.num_ranks)]

    def _set_train(self) -> None:
        self.train, self.valid = True, False

    def _set_valid(self) -> None:
        self.train, self.valid = False, True

    def change_task(self) -> None:
        self._set_valid() if self.train else self._set_train()
        self._reset()

    def finish_task(self, rank: int) -> None:
        self.completed[rank] = True

    def finished_task(self) -> bool:
        return all(self.completed)


class Dashboard:
    def __init__(self, cfg: DictConfig, queue: mp.Queue) -> None:
        self.cfg = cfg
        self.queue = queue

        # get values that we be reused
        self.num_epochs = self.cfg.train.epochs
        self.num_ranks = self.cfg.cuda.num_gpus

        # configure TaskProgressColumn and MofNCompleteColumn to be equal width
        self.separator = "/"
        self.min_width = max(
            len("100%"), (len(str(self.num_epochs) * 2) + len(self.separator))
        )

        self._create_epochs_progress()
        self._create_rank_pbars()

    def _create_epochs_progress(self) -> Progress:
        self.epochs_progress = Progress(
            "{task.description}",
            BarColumn(bar_width=5),
            MofNCompleteColumn(
                table_column=Column(
                    justify="right",
                    min_width=self.min_width,
                ),
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self.epochs_progress.add_task("Epoch", total=self.num_epochs)

    def _create_rank_pbars(self) -> Progress:

        self.rank_pbars = {i: None for i in range(self.num_ranks)}

        for rank in self.rank_pbars.keys():

            pbar = Progress(
                "{task.description}",
                BarColumn(bar_width=5, pulse_style="bar.back"),
                TaskProgressColumn(
                    justify="right",
                    table_column=Column(min_width=self.min_width),
                ),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )

            pbar.add_task("Train")
            pbar.add_task("Valid", start=False)

            self.rank_pbars[rank] = pbar

    def read_message(
        self,
        message: Tuple[int, int, int, int] | int,
        epoch: EpochMonitor,
    ) -> None:

        # ignore sentinel messages
        if message == -1:
            return

        rank, count, subtotal, current_epoch = message

        # advance the current task's progress bar for the specified rank
        self.rank_pbars[rank].update(
            epoch.task, completed=count, total=subtotal
        )

        # keep track of when batches are finished for each rank
        if count == subtotal:
            epoch.finish_task(rank=rank)

        # update the overall progress bar
        if epoch.finished_task():

            epoch.change_task()

            for rank, pbar in self.rank_pbars.items():
                pbar.start_task(epoch.task)

            # advance epoch progress at the beginning of each train loop
            if epoch.train:

                self.epochs_progress.advance(task_id=0, advance=1)

                # reset subprocess bars and the end of the epoch
                if current_epoch != self.num_epochs:
                    for rank, pbar in self.rank_pbars.items():
                        pbar.reset(0, completed=0, total=1)
                        pbar.reset(1, completed=0, total=1, start=False)

    def show(self) -> None:

        # TODO: use layout instead of grid
        g = Table.grid()

        with Live(g, refresh_per_second=10, transient=False):

            g.add_row(Panel.fit(self.epochs_progress, padding=(1, 2)))
            row = []
            for i, rank in self.rank_pbars.items():
                panel = Panel.fit(rank, title=f"[b]Rank {i}", padding=(1, 2))
                row.append(panel)
                if i % 2 == 1:
                    g.add_row(*row)
                    row = []

            # keep track of the current state of training
            epoch = EpochMonitor(self.num_ranks)

            while not self.epochs_progress.finished:

                # subprocesses can push an arbitrary message into a queue, which
                # is queried from the main process until all tasks are completed
                self.read_message(message=self.queue.get(), epoch=epoch)
