import logging
import os
from logging.handlers import QueueHandler, QueueListener
from typing import Callable

import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group

from .progress import Dashboard

logger = logging.getLogger(__name__)


def worker(
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
    logger.root.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    torch.distributed.barrier()
    logger.info("Logging from rank 0 only")

    # execute the target function from its own subprocess
    dist_fn(**locals())

    # terminate DDP worker after function has completed
    destroy_process_group()


def ddp(dist_fn: Callable, cfg: DictConfig) -> None:
    """
    Configure logging and progress bar(s) in a process-safe way, then create
    a separate subprocess for each GPU to run the specified function.
    """

    mp.set_start_method("spawn")

    # share memory between subprocesses by adding items to a queue
    queue = mp.Queue()

    # set up a separate queue to handle log messages from subprocesses
    log_queue = mp.Queue()
    listener = QueueListener(log_queue, *logger.root.handlers)
    listener.start()

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
        p = mp.Process(target=worker, kwargs=kwargs)
        p.start()
        subprocesses.append(p)

    # progress bars are displayed from the main process;
    # wait until the queue is populated before starting
    while queue.empty():
        continue

    # display progress bars
    dashboard = Dashboard(cfg, queue)
    dashboard.show()

    # wait for all subprocesses to finish together
    for p in subprocesses:
        p.join()

    # log_queue.put_nowait(None)
    listener.stop()