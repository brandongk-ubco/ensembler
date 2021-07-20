import torch
from torch import nn
from functools import partial


class LiHT(nn.Module):
    def __init__(self, alpha1=0., alpha2=0., center=0., width=0.5):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.center = center
        self.width = width
        self.half_width = self.width / 2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.center - self.half_width
        x2 = self.center + self.half_width

        output = torch.empty_like(input)

        range1 = input < x1
        range2 = input > x2
        range3 = torch.logical_and(input <= x2, input >= x1)

        b1 = -(1 + self.alpha1 * x1)
        output[range1] = self.alpha1 * input[range1] + b1

        b2 = 1 - self.alpha1 * x2
        output[range2] = self.alpha2 * input[range2] + b2

        m3 = 1 / self.half_width
        b3 = self.center + m3 * self.center
        output[range3] = m3 * input[range3] + b3

        return output


PWLinear = partial(LiHT, alpha1=0.5, alpha2=0.5)
