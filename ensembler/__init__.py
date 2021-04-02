from enum import Enum
from ensembler import train
from ensembler import dataset_statistics


class Tasks(Enum):
    train = "train"
    dataset_statistic = "dataset_statistics"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    def description(task):
        if task == "train":
            return train.description
        if task == "dataset_statistics":
            return dataset_statistics.description

        raise ValueError("Task {} not defined".format(task))

    def add_argparse_args(task):
        if task == "train":
            return train.add_argparse_args
        if task == "dataset_statistics":
            return dataset_statistics.add_argparse_args

        raise ValueError("Task {} not defined".format(task))

    def get(task):
        if task == "train":
            return train.execute
        if task == "dataset_statistics":
            return dataset_statistics.execute

        raise ValueError("Task {} not defined".format(task))
