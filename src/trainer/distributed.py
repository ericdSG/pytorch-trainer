import logging
import os
from logging.handlers import QueueHandler
from typing import Callable

import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group

from .progress import process_queue

logger = logging.getLogger(__name__)


def worker(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
    **kwargs,
) -> None:
    """
    A worker subprocess. Not intended for DDP: use ddp_worker()
    """

    # subprocess logs are sent to log_queue and managed by main processes
    logger.root.addHandler(QueueHandler(log_queue))
    logger.info(f"Initialized worker")

    # execute the target function
    dist_fn(**locals())


def ddp_worker(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
    rank: int,
    **kwargs,
) -> None:

    # configure current worker within the DistributedDataParallel context
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=cfg.cuda.num_gpus)

    # subprocess logs are sent to log_queue and managed by main processes
    logger.root.addHandler(QueueHandler(log_queue))
    logger.info(f"Initialized DDP rank {rank}")
    logger.root.setLevel(logging.root.level if rank == 0 else logging.WARNING)
    torch.distributed.barrier()
    logger.info("Logging from rank 0 only")

    # execute the target function
    dist_fn(**locals())

    # terminate DDP worker after function has completed
    destroy_process_group()


def spawn(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
) -> None:
    """
    Create a single subprocess (worker) to execute the dist_fn.
    """

    # breakpoints are only supported for single-process applications
    if cfg.debug:
        dist_fn(**locals())
        return

    # create a dedicated subprocess for the target function
    p = mp.Process(target=worker, kwargs=locals())
    p.start()

    # stdout (logging and progress bars) is handled from main process
    process_queue(cfg, queue)


def ddp_spawn(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
) -> None:
    """
    Create a separate subprocess for each GPU to run the specified function.
    """

    kwargs = locals()
    subprocesses = []  # keep track of subprocesses created for each GPU

    # spawn a subprocess for each GPU
    for rank in range(cfg.cuda.num_gpus):
        p = mp.Process(target=ddp_worker, kwargs=kwargs | dict(rank=rank))
        p.start()
        subprocesses.append(p)

    # stdout (logging and progress bars) is handled from main process
    process_queue(cfg, queue)

    # wait for all subprocesses to finish together
    for p in subprocesses:
        p.join()
