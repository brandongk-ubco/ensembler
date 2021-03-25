from argparse import ArgumentParser
# from pytorch_lightning import Trainer
from ensembler.datasets import Datasets
from ensembler import Tasks
import os

parser = ArgumentParser()
subparsers = parser.add_subparsers(title="task", dest="task")
# parser = Trainer.add_argparse_args(parser)
parser.add_argument(dest="dataset", type=str, choices=Datasets.choices())
parser.add_argument('--data_dir',
                    type=str,
                    nargs='?',
                    const=os.environ.get("DATA_DIR", None),
                    default=os.environ.get("DATA_DIR", None))
for task_name in Tasks.choices():
    subparser = subparsers.add_parser(task_name,
                                      help=Tasks.description(task_name))
    subparser = Tasks.add_argparse_args(task_name)(subparser)

args = parser.parse_args()
