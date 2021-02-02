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
                             nn.BatchNorm2d(self.width), self.activation())


class UNetEncoder(nn.Module, EncoderMixin):
    def __init__(self,
                 width=10,
                 depth=7,
                 block=UNetBlock,
                 activation=nn.ReLU,
                 **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._width = width
        self._in_channels = 3
        self.activation = activation
        self.block = block

        self.conv1 = nn.Conv2d(self._in_channels, self._width, 1).to("cuda")

        self.blocks = [
            self.block(self._width, self.activation).to("cuda")
            for b in range(self._depth * 2)
        ]

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return [self._width] * (self._depth + 1)

    def get_stages(self):
        stages = [self.conv1]

        for i in range(0, self._depth * 2, 2):
            stages.append(
                nn.Sequential(self.blocks[i], self.blocks[i + 1],
                              nn.MaxPool2d(2)))

        return stages

    def forward(self, x):
        features = []
        for stage in self.get_stages():
            x = stage(x)
            features.append(x)

        return features

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("Dilated mode not supported!")


for activation_name in Activations.choices():
    activation = Activations.get(activation_name)
    encoders.update({
        "{}_unet".format(activation_name): {
            "encoder": UNetEncoder,
            "pretrained_settings": [],
            'params': {
                "activation": activation
            }
        }
    })
