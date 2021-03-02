import matplotlib
import numpy as np
import os
from p_tqdm import p_uimap as mapper
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import glob
from parameters import args
from datasets import Datasets

matplotlib.use('Agg')

outdir = "/mnt/d/work/repos/ensembler/lightning_logs/version_2/predictions/"


def to_one_hot(mask, num_classes):

    label_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes),
                          dtype=np.float32)

    for clazz in range(num_classes):
        label_mask[:, :, clazz][mask == clazz] = 1

    return label_mask


def evaluate_prediction(src):
    prediction = np.load(src)
    predicted_mask = prediction["prediction"].transpose(1, 2, 0)
    image = prediction["image"].transpose(1, 2, 0)
    mask = prediction["mask"].transpose(1, 2, 0)

    filename = os.path.basename(src)
    name, ext = os.path.splitext(filename)
    mask = to_one_hot(np.argmax(mask, axis=2), dataset.num_classes)
    predicted_mask = to_one_hot(np.argmax(predicted_mask, axis=2),
                                dataset.num_classes)

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
        elif class_exists and not prediction_exists:
            actual_f1_score = 0.0
            actual_jaccard_score = 0.0
            actual_recall_score = 0.0
            actual_precision_score = 0.0
        elif not class_exists and prediction_exists:
            actual_f1_score = 0.0
            actual_jaccard_score = 0.0
            actual_recall_score = 1.0
            actual_precision_score = 0.0
        elif not class_exists and not prediction_exists:
            actual_f1_score = 1.0
            actual_jaccard_score = 1.0
            actual_recall_score = 1.0
            actual_precision_score = 1.0

        result = result.append(
            {
                "class": class_name,
                "image": name,
                "precision": actual_precision_score,
                "recall": actual_recall_score,
                "f1_score": actual_f1_score,
                "iou": actual_jaccard_score
            },
            ignore_index=True)
    return result


if __name__ == '__main__':

    dict_args = vars(args)

    dataset = Datasets.get(dict_args["dataset"])

    predictions = glob.glob(os.path.join(outdir, "*.npz"))

    df = pd.DataFrame()
    for result in mapper(evaluate_prediction, predictions):
        df = df.append(result, ignore_index=True)
    df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    means = df.groupby(by=["class"]).mean().reset_index()
    means.to_csv(os.path.join(outdir, "mean_metrics.csv"), index=False)
