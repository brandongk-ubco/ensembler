import numpy as np
import os
from ensembler.p_tqdm import p_uimap as mapper
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
import glob
import yaml
from ensembler.datasets import Datasets
from functools import partial

description = "Evaluate the performance of a model."


def add_argparse_args(parser):
    parser.add_argument('version', type=str)
    return parser


def to_one_hot(mask, num_classes):

    label_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes),
                          dtype=np.float32)

    for clazz in range(num_classes):
        label_mask[:, :, clazz][mask == clazz] = 1

    return label_mask


def evaluate_prediction(src, dataset):
    prediction = np.load(src)
    predicted_mask = prediction["predicted_mask"]
    mask = prediction["mask"]

    filename = os.path.basename(src)
    name, ext = os.path.splitext(filename)
    mask = to_one_hot(mask, dataset.num_classes)
    predicted_mask = to_one_hot(predicted_mask, dataset.num_classes)

    result = pd.DataFrame()

    for i, class_name in enumerate(dataset.classes):
        class_mask = mask[:, :, i].astype(np.uint8)
        class_prediction = predicted_mask[:, :, i].astype(np.uint8)

        class_exists = np.max(class_mask) > 0
        prediction_exists = np.max(class_prediction) > 0

        if class_exists and prediction_exists:
            actual_f1_score = f1_score(class_mask,
                                       class_prediction,
                                       average="micro")
            actual_jaccard_score = jaccard_score(class_mask,
                                                 class_prediction,
                                                 average="micro")
            actual_recall_score = recall_score(class_mask,
                                               class_prediction,
                                               average="micro")
            actual_precision_score = precision_score(class_mask,
                                                     class_prediction,
                                                     average="micro")
            actual_precision_score = precision_score(class_mask,
                                                     class_prediction,
                                                     normalize=True)

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

        result = result.append(
            {
                "class": class_name,
                "image": name,
                "precision": actual_precision_score,
                "recall": actual_recall_score,
                "f1_score": actual_f1_score,
                "iou": actual_jaccard_score,
                "accuracy": actual_accuracy_score
            },
            ignore_index=True)
    return result


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

    predictions = glob.glob(os.path.join(predictions_dir, "*.npz"))

    dataset = Datasets.get(hparams["dataset_name"])

    df = pd.DataFrame()
    for result in mapper(partial(evaluate_prediction, dataset=dataset),
                         predictions):
        df = df.append(result, ignore_index=True)
    df.to_csv(os.path.join(model_dir, "metrics.csv"), index=False)

    means = df.groupby(by=["class"]).mean().reset_index()
    means.to_csv(os.path.join(model_dir, "mean_metrics.csv"), index=False)
