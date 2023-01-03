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

    # subprocess logs are sent to log_queue and managed by main processes
    if mp.current_process().name != "MainProcess":
        logger.root.addHandler(QueueHandler(log_queue))

    logger.info(f"Initialized worker")

    # execute the target function
    dist_fn(**locals())


def spawn(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
    **kwargs,
) -> None:
    """
    Create a single subprocess (worker) to execute the dist_fn so main process
    can handle logging and progress bars/dashboard.
    """

    # breakpoints are only supported from main process
    # dashboard will be disabled since it cannot run concurrently
    if cfg.debug:
        worker(**locals())
        return

    # create a dedicated subprocess for the target function
    p = mp.Process(target=worker, args=([dist_fn, cfg, queue, log_queue]))
    p.start()

    # stdout (logging and progress bars) is handled from main process
    process_queue(cfg, queue)


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


def ddp(
    dist_fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
) -> None:
    """
    Create a separate subprocess for each GPU to run the specified function.
    """

    subprocesses = []  # keep track of subprocesses created for each GPU

    for rank in range(cfg.cuda.num_gpus):

        # join function arguments with queues to pass to each subprocess
        kwargs = dict(
            cfg=cfg,
            dist_fn=dist_fn,
            queue=queue,
            log_queue=log_queue,
            rank=rank,
        )

        # spawn a subprocess for each GPU
        p = mp.Process(target=ddp_worker, kwargs=kwargs)
        p.start()
        subprocesses.append(p)

    # stdout (logging and progress bars) is handled from main process
    process_queue(cfg, queue)

    # wait for all subprocesses to finish together
    for p in subprocesses:
        p.join()
