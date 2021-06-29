from enum import Enum
from monai.networks.layers.factories import Act


class Activations(Enum):
    leaky_relu = "leaky_relu"
    relu = "relu"
    tanh = "tanh"
    swish = "swish"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(activation):
        if activation.value == "leaky_relu":
            return Act.LEAKYRELU
        if activation.value == "relu":
            return Act.RELU
        if activation.value == "tanh":
            return Act.TANH
        if activation.value == "swish":
            return Act.MEMSWISH

        raise ValueError("Activation %s not defined" % activation)
