from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.encoders import encoders
import torch.nn as nn
from Activations import Activations


class UNetBlock(nn.Sequential):
    def __init__(self, width, activation):
        super(UNetBlock, self).__init__()
        self.activation = activation
        self.width = width

        super(UNetBlock,
              self).__init__(nn.Conv2d(self.width, self.width, 3, padding=1),
                             nn.BatchNorm2d(self.width), self.activation(),
                             nn.Conv2d(self.width, self.width, 3, padding=1),
                             nn.BatchNorm2d(self.width), self.activation(),
                             nn.MaxPool2d(2))


class UNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, width, depth, activation, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth - 1
        self._width = width
        self.activation = activation
        self.set_in_channels(3)

        self.blocks = [
            UNetBlock(self._width, self.activation).to("cuda")
            for b in range(self._depth)
        ]

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return [self._width] * (self._depth + 1)

    def set_in_channels(self, in_channels):
        self._in_channels = in_channels
        self.conv1 = nn.Conv2d(self._in_channels, self._width, 1).to("cuda")

    def get_stages(self):
        return [self.conv1] + self.blocks

    def forward(self, x):
        features = []
        for stage in self.get_stages():
            x = stage(x)
            features.append(x)

        return features

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("Dilated mode not supported!")


for width in range(1, 61):
    for activation_name in Activations.choices():
        name = "unet_{}_width{}".format(activation_name, width)
        encoders.update({
            name: {
                "encoder": UNetEncoder,
                "pretrained_settings": [],
                'params': {
                    "activation": Activations.get(activation_name),
                    "width": width
                }
            }
        })
