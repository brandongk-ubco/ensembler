from ensembler.Model import Segmenter
import os
from ensembler.datasets import Datasets
import glob
from ensembler.train import get_augmenters, get_augments
import yaml
import pytorch_lightning as pl

description = "Predict using a trained model."


def add_argparse_args(parser):
    parser.add_argument('version', type=str)
    return parser


def execute(args):

    dict_args = vars(args)

    base_dir = os.path.abspath(".")

    model_dir = os.path.join(base_dir, "lightning_logs", dict_args["version"])

    hparams_file = os.path.join(model_dir, "hparams.yaml")
    with open(hparams_file, "r") as hf:
        hparams = yaml.load(hf)

    hparams.update(dict_args)

    checkpoints = glob.glob(
        os.path.join(model_dir, "checkpoints", "epoch=*.ckpt"))

    assert len(checkpoints) > 0
    checkpoint_scores = [
        dict([k.split("=") for k in os.path.split(c[:-5])[-1].split("-")])
        for c in checkpoints
    ]
    checkpoint_scores = [
        dict([(k, float(v)) for (k, v) in d.items()])
        for d in checkpoint_scores
    ]

    checkpoints = [{"path": c} for c in checkpoints]

    checkpoints = [{
        **checkpoint_scores[i],
        **checkpoints[i]
    } for i in range(len(checkpoints))]

    checkpoint = sorted(checkpoints,
                        key=lambda r: -r["val_loss"] * 1000 + r["val_iou"] *
                        100 + r["epoch"] / 100)[-1]["path"]

    print("Using Checkpoint: {}".format(checkpoint))

    dataset = Datasets.get(hparams["dataset_name"])

    train_data, val_data, test_data = dataset.get_dataloaders(
        os.path.join(hparams["data_dir"], hparams["dataset_name"]),
        get_augmenters(hparams["batch_loss_multiplier"] > 0),
        hparams["batch_size"],
        get_augments(hparams["patch_height"], hparams["patch_width"]))

    hparams["dataset"] = dataset
    hparams["train_data"] = train_data
    hparams["val_data"] = val_data
    hparams["test_data"] = test_data

    m = Segmenter.load_from_checkpoint(checkpoint, **hparams)

    trainer = pl.Trainer(gpus=1)
    trainer.test(m)
