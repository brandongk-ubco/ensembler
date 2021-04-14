import pytorch_lightning as pl
import sys
import os
from ensembler.Model import Segmenter as model
from ensembler.augments import get_augments
from ensembler.callbacks import RecordTrainStatus
from ensembler.datasets.AugmentedDataset import RepeatedDatasetAugmenter, DatasetAugmenter
from ensembler.datasets import Datasets

description = "Train a model."


def add_argparse_args(parser):
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() - 1),
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset_split_seed', type=int, default=42)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--patch_height', type=int, default=512)
    parser.add_argument('--patch_width', type=int, default=512)
    parser = model.add_model_specific_args(parser)
    return parser


def get_augmenters():
    return RepeatedDatasetAugmenter, DatasetAugmenter


def execute(args):

    dict_args = vars(args)

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss',
                                   patience=3 * dict_args["patience"]),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        RecordTrainStatus()
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1,
        callbacks=callbacks,
        min_epochs=dict_args["patience"],
        deterministic=True,
        max_epochs=sys.maxsize,
        accumulate_grad_batches=dict_args["accumulate_grad_batches"])

    dataset = Datasets.get(dict_args["dataset_name"])

    train_data, val_data, test_data = dataset.get_dataloaders(
        os.path.join(dict_args["data_dir"], dict_args["dataset_name"]),
        get_augmenters(), dict_args["batch_size"],
        get_augments(dict_args["patch_height"], dict_args["patch_width"]))

    trainer.fit(model(dataset, train_data, val_data, test_data, **dict_args))
