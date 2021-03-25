from argparse import ArgumentParser
# from pytorch_lightning import Trainer
from ensembler.datasets import Datasets
from ensembler import Tasks

parser = ArgumentParser()
subparsers = parser.add_subparsers(title="task", dest="task")
# parser = Trainer.add_argparse_args(parser)
parser.add_argument(dest="dataset", type=str, choices=Datasets.choices())
parser.add_argument('--dataset_split_seed', type=int, default=42)
parser.add_argument('--seed', type=int, default=42)
for task_name in Tasks.choices():
    subparser = subparsers.add_parser(task_name,
                                      help=Tasks.description(task_name))
    Tasks.add_argparse_args(task_name)(subparser)

args = parser.parse_args()
