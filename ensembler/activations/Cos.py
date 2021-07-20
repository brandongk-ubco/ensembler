import torch
from torch import nn


class Cos(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input - torch.cos(input)