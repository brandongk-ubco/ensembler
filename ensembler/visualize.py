import numpy as np
import os
from ensembler.p_tqdm import p_umap as mapper
from ensembler.train import get_augmenters, get_augments
import glob
import yaml
from ensembler.datasets import Datasets
from functools import partial
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')

description = "Visualize Predictions of a model."


def add_argparse_args(parser):
    parser.add_argument('version', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)
    return parser


def visualize(outfile, image, mask, predicted_mask):
    num_classes = mask.shape[2]
    intensity = 255 // (num_classes + 1)
    mask_img = np.argmax(mask, axis=2) * intensity
    predicted_mask_img = np.argmax(predicted_mask, axis=2) * intensity

    fig, axs = plt.subplots(3, 1)

    if image.shape[2] == 1:
        axs[0].imshow(image.squeeze(), cmap="gray")
    else:
        axs[0].imshow(image)

    axs[1].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
    axs[2].imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)

    for ax_i in axs:
        ax_i.axis('off')

    plt.savefig(outfile)
    plt.close()


def visualize_prediction(idx, loader, image_names, predictions_dir, threshold,
                         dataset, reducer):

    image, mask = loader[idx]
    image = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
    mask = mask.clone().detach().cpu().numpy().transpose(1, 2, 0)
    image_name = image_names[idx // 4]
    predictions = glob.glob(
        os.path.join(predictions_dir, "{}*.npz".format(image_name)))

    predicted_results = []

    for prediction_file in predictions:
        prediction = np.load(prediction_file)
        predicted_mask = prediction["predicted_mask"].astype(mask.dtype) / 255
        predicted_results.append(predicted_mask)

        filename = os.path.basename(prediction_file)
        name, ext = os.path.splitext(filename)
        view = name.split("_")[-1]
        name = "_".join(name.split("_")[:-1])

        out_image = os.path.join(predictions_dir,
                                 "{}_{}.png".format(name, view))
        visualize(out_image, image, mask, predicted_mask)

    if len(predicted_results) > 0:
        combined = np.stack(predicted_results)
        predicted_mask = reducer(combined)

        out_image = os.path.join(predictions_dir, "{}.png".format(image_name))

        visualize(out_image, image, mask, predicted_mask)


def get_reducer(batch_loss):
    if batch_loss:
        print("Using noisy-or aggregation")
        return lambda y: 1 - np.prod(1 - y, axis=0)
    else:
        print("Using average aggregation")
        return lambda y: np.mean(y, axis=0)


def execute(args):

    dict_args = vars(args)

    base_dir = os.path.abspath(".")

    model_dir = os.path.join(base_dir, "lightning_logs", dict_args["version"])

    predictions_dir = os.path.join(model_dir, "predictions")

    hparams_file = os.path.join(model_dir, "hparams.yaml")
    with open(hparams_file, "r") as hf:
        hparams = yaml.load(hf)

    hparams.update(dict_args)

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

    samples = list(range(0, len(test_data), 4))
    image_names = test_data.dataset.get_image_names()
    reducer = get_reducer(hparams["batch_loss_multiplier"] > 0.)

    print("Visualizing Predictions")

    mapper(partial(visualize_prediction,
                   loader=test_data,
                   image_names=image_names,
                   predictions_dir=predictions_dir,
                   threshold=hparams["threshold"],
                   reducer=reducer,
                   dataset=dataset),
           samples,
           num_cpus=dict_args["num_workers"])
