from enum import Enum
import datasets.voc as voc
import datasets.cityscapes as cityscapes
import datasets.wrinkler as wrinkler
import datasets.severstal as severstal


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