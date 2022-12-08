import random

import numpy as np
import torch


def param_count(module: torch.nn.Module) -> int:
    return sum([p.data.nelement() for p in module.parameters()])


def set_global_seed(seed: int) -> None:
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
