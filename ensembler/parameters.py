from argparse import ArgumentParser
from ensembler.datasets import Datasets
from ensembler import Tasks
import os


def parse_parameters():

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="task", dest="task")
    for task_name in Tasks.choices():
        subparser = subparsers.add_parser(task_name,
                                          help=Tasks.description(task_name))
        subparser = Tasks.add_argparse_args(task_name)(subparser)

    return parser.parse_args()
