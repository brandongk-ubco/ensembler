from enum import Enum
from ensembler.train import execute as train
from ensembler.dataset_statistics import execute as dataset_statistics


class Tasks(Enum):
    train = "train"
    dataset_statistic = "dataset-statistics"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def get(task):
        if task == "train":
            return train
        if task == "dataset-statistics":
            return dataset_statistics

        raise ValueError("Task {} not defined".format(task))
