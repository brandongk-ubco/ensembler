from argparse import ArgumentParser
from Model import Segmenter
# from pytorch_lightning import Trainer
from datasets import Datasets

parser = ArgumentParser()
parser = Segmenter.add_model_specific_args(parser)
# parser = Trainer.add_argparse_args(parser)
parser.add_argument('--dataset', type=str, choices=Datasets.choices())
parser.add_argument('--dataset_split_seed', type=int, default=42)
args = parser.parse_args()
