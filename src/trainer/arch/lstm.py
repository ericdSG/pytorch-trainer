import logging

import torch

from .modules import param_count

logger = logging.getLogger(__name__)


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool = True,  # (batch, seq, feat) instead of (s, b, f)
        dropout: float = 0.0,  # PyTorch default
        bidirectional: bool = False,  # PyTorch default
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.linear = torch.nn.Linear(
            in_features=hidden_size * 2 if bidirectional else hidden_size,
            out_features=out_features,
        )

        name = self.__class__.__name__
        logger.debug(f"{name} parameters = {param_count(self):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # assign tensors to adjacent memory blocks to optimize LSTM unrolling
        x = x.contiguous()

        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = self.linear(x)

        return x
