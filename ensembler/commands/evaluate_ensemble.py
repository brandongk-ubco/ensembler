from ensembler.Dataset import Dataset
from ensembler.datasets import Datasets
from argh import arg
import os
import numpy as np
from ensembler.utils import classwise
from ensembler.p_tqdm import p_uimap as mapper
from functools import partial
import pandas as pd
from typing import List
import hashlib
import json

metrics = {
    "iou":
        lambda y_hat, y: -1 if not y.max() > 0 else (y_hat * y).sum() /
        (y_hat + y).astype(bool).sum(),
    "recall":
        lambda y_hat, y: -1 if not y.max() > 0 else (y_hat * y).sum() / y.sum(),
    "precision":
        lambda y_hat, y: -1 if not y.max() > 0 else 0
        if not y_hat.max() > 0 else (y_hat * y).sum() / y_hat.sum()
}


def load_predictions(image_name, prediction_dir, job_hashes, dtype):
    predictions = []
    for job_hash in job_hashes:
        prediction_file = os.path.join(prediction_dir, job_hash, "predictions",
                                       "{}.npz".format(image_name))
        if not os.path.exists(prediction_file):
            raise ValueError(
                "Cannot find prediction for job {} image {}.  Looked at {}".
                format(job_hash, image_name, prediction_file))
        prediction = np.load(prediction_file)["predicted_mask"].astype(
            dtype) / 255
        predictions.append(prediction)
    return np.stack(predictions, axis=-1)


def process_sample(ground_truth, threshold, classes, prediction_loader):
    (image, mask), image_name = ground_truth

    image = np.moveaxis(np.array(image).squeeze(0), 0, -1)
    mask = np.moveaxis(np.array(mask).squeeze(0), 0, -1)
    prediction = prediction_loader(image_name)
    prediction = np.average(prediction, axis=-1)
    prediction = np.where(prediction > threshold, 1., 0.)

    results = []

    for metric, metric_func in metrics.items():
        result = classwise(prediction, mask, metric=metric_func)

        results_list = list(zip(classes, result))
        for clazz, result in results_list:
            if result > -1:
                results.append({
                    "image": image_name,
                    "class": clazz,
                    "threshold": threshold,
                    "metric": metric,
                    "value": result
                })
    return results


@arg('dataset', choices=Datasets.choices())
@arg('job_hashes', nargs='+')
def evaluate_ensemble(dataset: Datasets,
                      base_dir: str,
                      job_hashes: List[str],
                      threshold: float = 0.5):

    job_hashes.sort()

    prediction_dir = os.path.abspath(base_dir)

    dataset = Datasets[dataset]
    datamodule = Dataset(dataset=dataset, batch_size=1)
    dataset = Datasets.get(dataset.value)
    dataloader = datamodule.test_dataloader()

    image_names = datamodule.test_data.dataset.get_image_names()

    results = []

    loader = partial(load_predictions,
                     prediction_dir=prediction_dir,
                     job_hashes=job_hashes,
                     dtype=np.float32)

    func = partial(process_sample,
                   prediction_loader=loader,
                   threshold=threshold,
                   classes=dataset.classes)

    for sample_results in mapper(
            func,
            zip(dataloader, image_names),
            total=len(image_names),
            num_cpus=4,
    ):
        results += sample_results

    ensemble_hash = hashlib.md5("".join(job_hashes).encode()).hexdigest()
    ensemble_dir = os.path.join(base_dir, "ensembles", ensemble_hash)
    os.makedirs(ensemble_dir, exist_ok=True)

    outfile = os.path.join(os.path.abspath(ensemble_dir), "job_hashes.json")
    with open(outfile, "w") as jsonfile:
        json.dump(job_hashes, jsonfile, indent=4)

    outfile = os.path.join(os.path.abspath(ensemble_dir), "metrics.csv")
    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
