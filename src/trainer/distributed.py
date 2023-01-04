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


def worker(fn: Callable, **kwargs) -> None:
    """
    A worker subprocess (not intended for DDP)
    """

    # subprocess logs are sent to log_queue and managed by main processes
    logger.root.addHandler(QueueHandler(kwargs["log_queue"]))
    logger.info(f"Initialized worker")
    assert not kwargs["cfg"].cuda.ddp

    # execute the target function
    fn(**kwargs)


def ddp_worker(fn: Callable, rank: int, world_size: int, **kwargs) -> None:
    """
    A worker subprocess with setup/teardown code for DDP
    """

    # configure current worker within the DistributedDataParallel context
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert kwargs["cfg"].cuda.ddp and torch.distributed.is_initialized()

    # subprocess logs are sent to log_queue and managed by main processes
    logger.root.addHandler(QueueHandler(kwargs["log_queue"]))
    logger.info(f"Initialized DDP rank {rank}")
    logger.root.setLevel(logging.root.level if rank == 0 else logging.WARNING)
    torch.distributed.barrier()
    logger.info("Logging from rank 0 only")

    # execute the target function
    fn(**locals() | kwargs)

    # terminate DDP worker after function has completed
    destroy_process_group()


def spawn(
    fn: Callable,
    cfg: DictConfig,
    queue: mp.Queue,
    log_queue: mp.Queue,
) -> None:
    """
    Create a subprocess (worker) for each GPU
    """

    if cfg.debug:  # disable multiprocessing to use breakpoints
        fn(**locals())
        return

    if cfg.cuda.ddp:
        world_size = cfg.cuda.num_gpus
        target = ddp_worker
    else:  # single-GPU or DataParallel
        world_size = 1
        target = worker

    kwargs = locals()  # freeze locals before creating temp variables in loop
    subprocesses = []  # keep track of subprocesses created for each GPU

    # spawn a subprocess for each GPU
    for rank in range(world_size):
        kwargs = kwargs | dict(rank=rank, world_size=world_size)
        p = mp.Process(target=target, kwargs=kwargs)
        p.start()
        subprocesses.append(p)

    # stdout (logging and progress bars) is handled from main process
    process_queue(cfg, queue)

    # wait for all subprocesses to finish together
    for p in subprocesses:
        p.join()
