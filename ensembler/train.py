import pytorch_lightning as pl
import sys
import os
from ensembler.Model import Segmenter as model
from ensembler.augments import get_augments
from ensembler.callbacks import RecordTrainStatus, WandbFileUploader
from ensembler.datasets.AugmentedDataset import RepeatedDatasetAugmenter, DatasetAugmenter, RepeatedBatchDatasetAugmenter, BatchDatasetAugmenter
from ensembler.datasets import Datasets
from pytorch_lightning.loggers import WandbLogger

description = "Train a model."


def add_argparse_args(parser):
    parser.add_argument('--batch_loss_multiplier', type=float, default=0.)
    parser.add_argument('--base_loss_multiplier', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers',
                        type=int,
                        default=os.environ.get("NUM_WORKERS",
                                               os.cpu_count() - 1)),
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataset_split_seed', type=int, default=42)
    parser.add_argument('--accumulate_grad_batches', type=int, default=3)
    parser.add_argument('--patch_height', type=int, default=512)
    parser.add_argument('--patch_width', type=int, default=512)
    parser = model.add_model_specific_args(parser)
    return parser


def get_augmenters(batch_loss):
    if batch_loss:
        return RepeatedBatchDatasetAugmenter, BatchDatasetAugmenter, BatchDatasetAugmenter
    else:
        return RepeatedDatasetAugmenter, DatasetAugmenter, BatchDatasetAugmenter


def execute(args):

    dict_args = vars(args)

    callbacks = [
        pl.callbacks.EarlyStopping(patience=3 * dict_args["patience"],
                                   monitor='val_loss',
                                   mode='min'),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode="min",
            filename='{epoch}-{val_loss:.6f}-{val_iou:.3f}'),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode="min",
            filename='{epoch}-{val_loss:.6f}-{val_iou:.3f}'),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode="min",
            filename='weights-{epoch}-{val_loss:.6f}-{val_iou:.3f}',
            save_weights_only=True),
        RecordTrainStatus(),
        WandbFileUploader(["*.png", "trainer.json"])
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    wandb_logger = WandbLogger(project=dict_args["dataset_name"],
                               entity='acislab',
                               name='efficientnet-b0-adaptive-tversky-loss')

    dataset = Datasets.get(dict_args["dataset_name"])

    train_data, val_data, test_data = dataset.get_dataloaders(
        os.path.join(dict_args["data_dir"], dict_args["dataset_name"]),
        get_augmenters(dict_args["batch_loss_multiplier"] > 0),
        dict_args["batch_size"],
        get_augments(dict_args["patch_height"], dict_args["patch_width"]))

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1,
        callbacks=callbacks,
        min_epochs=dict_args["patience"],
        deterministic=True,
        max_epochs=sys.maxsize,
        accumulate_grad_batches=dict_args["accumulate_grad_batches"],
        logger=wandb_logger,
        move_metrics_to_cpu=True,
        limit_train_batches=min(len(train_data), 2000))

    trainer.fit(model(dataset, train_data, val_data, test_data, **dict_args))
