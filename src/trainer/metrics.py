from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class AverageMeter:
    """
    Computes and stores the average and current value.
    All values must be (detached) cuda tensors so DDP can sync after each epoch.
    """

    def __init__(
        self,
        name: torch.nn.Module,
        device: int,
    ) -> None:
        self.name = name.__class__.__name__
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.val = torch.Tensor([0.0]).to(self.device)
        # self.avg = torch.Tensor([0.0]).to(self.device)
        self.sum = torch.Tensor([0.0]).to(self.device)
        self.count = torch.Tensor([0.0]).to(self.device)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count

    def __repr__(self) -> str:
        fields = [
            f"name={self.name}",
            # f"avg={self.avg.item():0.5f}",
            f"count={int(self.count.item())}",
            f"sum={self.sum.item():0.5f}",
            f"val={self.val.item():0.5f}",
        ]
        return f"AverageMeter({', '.join(fields)})"
