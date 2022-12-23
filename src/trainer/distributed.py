import os
from typing import Callable

import torch.multiprocessing as mp
from omegaconf import DictConfig
from rich import progress
from torch.distributed import destroy_process_group, init_process_group


def ddp_setup(
    rank: int,
    dist_fn: Callable,
    queue: mp.Queue,
    cfg: DictConfig,
    **kwargs,
) -> None:

    # configure current worker within the DistributedDataParallel context
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=cfg.cuda.num_gpus)

    # # reset file handler in append mode for each subprocess
    # file_handler = logging.FileHandler(filename=cfg.log, mode="a")
    # # get format from dummy NullHandler
    # file_handler.setFormatter(logging.root.handlers[-1].formatter)
    # file_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    # logging.getLogger().addHandler(file_handler)
    # logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    # begin training separately on each GPU (rank)
    dist_fn(**locals())

    # terminate DDP worker after function has completed
    destroy_process_group()


def train_ddp(cfg: DictConfig, dist_fn: Callable) -> None:
    """
    Configure logging and progress bar(s) in a process-safe way, then create
    a separate subprocess for each GPU to run the specified function.
    """

    mp.set_start_method("spawn", force=True)

    # share memory between subprocesses
    queue = mp.Queue()

    # configure the progess bar
    with progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        progress.TimeRemainingColumn(),
        progress.TimeElapsedColumn(),
        refresh_per_second=1,
        transient=True,
    ) as pbar:

        # keep track of subprocesses created for each GPU
        processes = []
        for rank in range(cfg.cuda.num_gpus):

            # create a progress bar for each subprocess; task ID = GPU rank
            task_id = pbar.add_task(f"GPU {rank}", visible=True)

            # join function arguments with queue to pass to each subprocess
            kwargs = dict(cfg=cfg, dist_fn=dist_fn, queue=queue, rank=rank)

            # spawn a subprocess for each GPU
            p = mp.Process(target=ddp_setup, kwargs=kwargs)
            p.start()
            processes.append(p)

        # keep track of when each GPU finishes its task an epoch
        tasks_completed = {
            e: [0 for _ in processes] for e in range(1, cfg.train.epochs + 1)
        }

        # subprocesses push a tuple of information into the queue, which can be
        # queried from the main process on a loop until all tasks are completed
        while True:

            task_id, count, subtotal, current_epoch = queue.get()
            pbar.update(task_id, completed=count, total=subtotal)

            if count == subtotal:
                tasks_completed[current_epoch][task_id] = True

            if all(tasks_completed[cfg.train.epochs]):
                break

        # wait for all processes to finish together
        for p in processes:
            p.join()
