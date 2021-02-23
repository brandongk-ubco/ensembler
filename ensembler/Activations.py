from enum import Enum
import torch.nn as nn


class Activations(Enum):
    leaky_relu = "leaky_relu"
    relu = "relu"
    elu = "elu"
    linear = "linear"
    swish = "swish"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(activation):
        if activation == "leaky_relu":
            return nn.LeakyReLU
        if activation == "relu":
            return nn.ReLU
        if activation == "elu":
            return nn.ELU
        if activation == "linear":
            return nn.Identity
        if activation == "swish":
            return nn.SiLU

        raise ValueError("Activation %s not defined" % activation)
