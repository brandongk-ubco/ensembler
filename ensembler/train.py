import pytorch_lightning as pl
import sys
import os
from ensembler.Model import Segmenter as model
from ensembler.augments import get_augments
from ensembler.callbacks import RecordTrainStatus

description = "Train a model."


def add_argparse_args(parser):
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataset_split_seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    parser = model.add_model_specific_args(parser)
    return parser


def execute(args):

    dict_args = vars(args)

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss',
                                   patience=2 * dict_args["patience"]),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        RecordTrainStatus()
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=1,
                                            callbacks=callbacks,
                                            min_epochs=dict_args["patience"],
                                            deterministic=True,
                                            max_epochs=sys.maxsize)

    trainer.fit(model(get_augments, **dict_args))
