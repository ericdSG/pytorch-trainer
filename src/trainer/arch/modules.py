"""
Collection of PyTorch nn.Modules for building models and utility functions
related to models.

Created: Nov 2022 by Piotr Ozimek
Updated: Dec 2022 by Eric DeMattos
"""

import torch


def param_count(module: torch.nn.Module) -> int:
    return sum([p.data.nelement() for p in module.parameters()])
