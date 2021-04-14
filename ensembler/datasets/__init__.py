from enum import Enum
import ensembler.datasets.voc as voc
import ensembler.datasets.wrinkler as wrinkler
import ensembler.datasets.severstal as severstal
import ensembler.datasets.cityscapes as cityscapes
import ensembler.datasets.initializers.cityscapes as cityscapes_initializer
from ensembler.datasets.helpers import *


class Datasets(Enum):
    voc = "voc"
    severstal = "severstal"
    wrinkler = "wrinkler"
    cityscapes = "cityscapes"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(dataset):
        if dataset == "voc":
            return voc
        if dataset == "severstal":
            return severstal
        if dataset == "wrinkler":
            return wrinkler
        if dataset == "cityscapes":
            return cityscapes

        raise ValueError("Dataset {} not defined".format(dataset))

    def get_initializer(dataset):
        if dataset == "voc":
            raise NotImplementedError
        if dataset == "severstal":
            raise NotImplementedError
        if dataset == "wrinkler":
            raise NotImplementedError
        if dataset == "cityscapes":
            return cityscapes_initializer

        raise ValueError("Dataset {} not defined".format(dataset))