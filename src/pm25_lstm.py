from __future__ import annotations

import torch
from torch import nn


class PM25LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(inputs)
        return self.output(encoded[:, -1, :]).squeeze(-1)
