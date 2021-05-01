import numpy as np
import os
from ensembler.p_tqdm import p_uimap as mapper
from ensembler.train import get_augmenters, get_augments
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
import glob
import yaml
from ensembler.datasets import Datasets
from functools import partial

description = "Evaluate the performance of a model."


def add_argparse_args(parser):
    parser.add_argument('version', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)
    return parser


def to_one_hot(mask, num_classes):

    label_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes),
                          dtype=np.float32)

    for clazz in range(num_classes):
        label_mask[:, :, clazz][mask == clazz] = 1

    return label_mask


def evaluate(y_hat, y):

    class_exists = np.max(y) > 0
    prediction_exists = np.max(y_hat) > 0

    if class_exists and prediction_exists:
        actual_f1_score = f1_score(y, y_hat, average="micro")
        actual_jaccard_score = jaccard_score(y, y_hat, average="micro")
        actual_recall_score = recall_score(y, y_hat, average="micro")
        actual_precision_score = precision_score(y, y_hat, average="micro")
        actual_accuracy_score = accuracy_score(y, y_hat, normalize=True)

    elif class_exists and not prediction_exists:
        actual_f1_score = 0.0
        actual_jaccard_score = 0.0
        actual_recall_score = 0.0
        actual_precision_score = 0.0
        actual_accuracy_score = 0.0
    elif not class_exists and prediction_exists:
        actual_f1_score = 0.0
        actual_jaccard_score = 0.0
        actual_recall_score = 1.0
        actual_precision_score = 0.0
        actual_accuracy_score = 0.0
    elif not class_exists and not prediction_exists:
        actual_f1_score = 1.0
        actual_jaccard_score = 1.0
        actual_recall_score = 1.0
        actual_precision_score = 1.0
        actual_accuracy_score = 1.0

    return {
        "precision": actual_precision_score,
        "recall": actual_recall_score,
        "f1_score": actual_f1_score,
        "iou": actual_jaccard_score,
        "accuracy": actual_accuracy_score,
        "in_class": class_exists
    }


def evaluate_prediction(idx, loader, image_names, predictions_dir, threshold,
                        dataset, reducer):
    image, mask = loader[idx]
    image = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
    mask = mask.clone().detach().cpu().numpy().transpose(1, 2, 0)
    image_name = image_names[idx // 4]
    predictions = glob.glob(
        os.path.join(predictions_dir, "{}*.npz".format(image_name)))

    predicted_results = []

    results = []
    for prediction_file in predictions:
        prediction = np.load(prediction_file)
        predicted_mask = prediction["predicted_mask"].astype(mask.dtype) / 255
        predicted_results.append(predicted_mask)

        filename = os.path.basename(prediction_file)
        name, ext = os.path.splitext(filename)
        view = name.split("_")[-1]
        name = "_".join(name.split("_")[:-1])

        for i, class_name in enumerate(dataset.classes):
            class_mask = mask[:, :, i]
            class_prediction = predicted_mask[:, :, i]
            class_prediction = np.where(class_prediction > threshold, 1,
                                        0).astype(np.uint8)

            class_mask = class_mask.astype(np.uint8)

            class_scores = evaluate(class_prediction, class_mask)
            class_scores["class"] = class_name
            class_scores["image"] = name
            class_scores["view"] = view

            results.append(class_scores)

    if len(predicted_results) > 0:
        combined = np.stack(predicted_results)
        predicted_mask = reducer(combined)

        for i, class_name in enumerate(dataset.classes):
            class_mask = mask[:, :, i]
            class_prediction = predicted_mask[:, :, i]
            class_prediction = np.where(class_prediction > threshold, 1,
                                        0).astype(np.uint8)

            class_mask = class_mask.astype(np.uint8)

            class_scores = evaluate(class_prediction, class_mask)
            class_scores["class"] = class_name
            class_scores["image"] = name
            class_scores["view"] = "combined"

            results.append(class_scores)

    return results


def get_reducer(batch_loss):
    if batch_loss:
        return lambda y: 1 - np.prod(1 - y, axis=0)
    else:
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

    rows = []
    for result in mapper(
            partial(
                evaluate_prediction,
                loader=test_data,
                image_names=image_names,
                predictions_dir=predictions_dir,
                threshold=hparams["threshold"],
                reducer=get_reducer(hparams["batch_loss_multiplier"] == 0.),
                dataset=dataset), samples):
        rows += result
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(model_dir, "metrics.csv"), index=False)

    means = df.groupby(by=["class", "view", "in_class"]).mean().reset_index()
    means.to_csv(os.path.join(model_dir, "mean_metrics.csv"), index=False)
