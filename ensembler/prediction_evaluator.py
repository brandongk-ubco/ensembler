import pytorch_lightning as pl
import matplotlib
from data import get_dataloaders
import numpy as np
import os
from p_tqdm import p_umap as mapper
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

pl.seed_everything(42)
matplotlib.use('Agg')

datapath = "/mnt/d/work/datasets/wrinkler/"
outdir = "/mnt/d/work/repos/wrinkler/lightning_logs/version_10/predictions/"
classes = ["background", "gripper", "wrinkle", "fabric"]


def evaluate_prediction(src):
    i, x, y = src
    name = os.path.splitext(
        (os.path.basename(test_loader.dataset.dataset.images[i])))[0]
    y_hat = np.load(os.path.join(outdir, "{}.npz".format(name)))["prediction"]
    predicted_mask = np.round(y_hat, 0)
    mask = y.squeeze(0).numpy()

    result = pd.DataFrame()

    for i, class_name in enumerate(classes):
        class_mask = mask[i, :, :].astype(np.uint8)
        class_prediction = predicted_mask[i, :, :].astype(np.uint8)
        result = result.append(
            {
                "class":
                class_name,
                "image":
                name,
                "precision":
                precision_score(class_mask, class_prediction, average="micro"),
                "recall":
                recall_score(class_mask, class_prediction, average="micro"),
                "f1_score":
                f1_score(class_mask, class_prediction, average="micro"),
                "iou":
                jaccard_score(class_mask, class_prediction, average="micro")
            },
            ignore_index=True)
    return result


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_dataloaders(datapath,
                                                            use_cache=False)

    df = pd.DataFrame()
    for result in mapper(evaluate_prediction,
                         [(i, x, y) for i, (x, y) in enumerate(test_loader)]):
        df = df.append(result, ignore_index=True)
    df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    means = df.groupby(by=["class"]).mean().reset_index()
    means.to_csv(os.path.join(outdir, "mean_metrics.csv"), index=False)
