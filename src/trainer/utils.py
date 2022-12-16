import random
from typing import Any

import numpy as np
import torch


def param_count(module: torch.nn.Module) -> int:
    return sum([p.data.nelement() for p in module.parameters()])


def set_global_seed(seed: int) -> None:
    """
    Doesn't really work, I fear.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def strip_module_prefix(checkpoint: dict[str, Any]) -> None:
    """
    Models trained with DDP add a "module." prefix to all state_dict values
    that is incompatible with non-DDP model objects, and vice-versa.
    """
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    for obj in ["model", "optimizer", "scheduler", "scaler"]:
        state_dict = checkpoint[f"{obj}_state"]
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
