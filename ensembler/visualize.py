import matplotlib
import numpy as np
import os
from matplotlib import pyplot as plt
from ensembler.p_tqdm import p_umap as mapper
import glob
import yaml
from ensembler.datasets import Datasets
from functools import partial

matplotlib.use('Agg')

description = "Visualize Predictions."


def add_argparse_args(parser):
    parser.add_argument('version', type=str)
    return parser


def visualize_prediction(src, outdir, num_classes):
    prediction = np.load(src)
    predicted_mask = prediction["predicted_mask"]
    image = prediction["image"]
    mask = prediction["mask"]

    filename = os.path.basename(src)
    name, ext = os.path.splitext(filename)

    intensity = 255 // num_classes
    mask_img = mask * intensity

    predicted_mask_img = predicted_mask * intensity

    fig, axs = plt.subplots(3, 1)

    if image.shape[2] == 1:
        axs[0].imshow(image.squeeze(), cmap="gray")
    else:
        axs[0].imshow(image)

    axs[1].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
    axs[2].imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)

    for ax_i in axs:
        ax_i.axis('off')

    plt.savefig(os.path.join(outdir, "{}.png".format(name)))
    plt.close()


def execute(args):

    dict_args = vars(args)

    base_dir = os.path.abspath(".")

    model_dir = os.path.join(base_dir, "lightning_logs",
                             "version_{}".format(dict_args["version"]))

    predictions_dir = os.path.join(model_dir, "predictions")

    hparams_file = os.path.join(model_dir, "hparams.yaml")
    with open(hparams_file, "r") as hf:
        hparams = yaml.load(hf)

    hparams.update(dict_args)

    dataset = Datasets.get(hparams["dataset_name"])

    predictions = glob.glob(os.path.join(predictions_dir, "*.npz"))

    mapper(
        partial(visualize_prediction,
                outdir=predictions_dir,
                num_classes=dataset.num_classes), predictions)
