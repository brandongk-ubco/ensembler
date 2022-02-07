from ensembler.Dataset import Dataset
from ensembler.datasets import Datasets
from argh import arg
import os
import numpy as np
from ensembler.utils import classwise
from ensembler.p_tqdm import p_uimap as mapper
from functools import partial
import pandas as pd

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


def process_sample(ground_truth, dataset, prediction_dir, threshold, classes):
    (image, mask), image_name = ground_truth

    val_dir = os.path.join(prediction_dir, "val")
    test_dir = os.path.join(prediction_dir, "test")

    if image_name in dataset.val_images:
        image_type = "val"
        indir = val_dir
    elif image_name in dataset.test_images:
        image_type = "test"
        indir = test_dir
    else:
        raise AttributeError(f"Unknown image set for {image_name}")

    prediction_file = os.path.join(indir, "{}.npz".format(image_name))
    if not os.path.exists(prediction_file):
        print("Can't find prediction for {}".format(image_name))
        return []

    image = np.moveaxis(np.array(image).squeeze(0), 0, -1)
    mask = np.moveaxis(np.array(mask).squeeze(0), 0, -1)
    prediction = np.load(prediction_file)["predicted_mask"].astype(
        mask.dtype) / 255
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
                    "type": image_type,
                    "value": result
                })
    return results


@arg('dataset', choices=Datasets.choices())
def evaluate(dataset: Datasets, base_dir: str, threshold: float = 0.5):
    prediction_dir = os.path.join(os.path.abspath(base_dir), "predictions")

    dataset = Datasets[dataset]
    datamodule = Dataset(dataset=dataset, batch_size=1)
    dataset = Datasets.get(dataset.value)
    dataloader = datamodule.test_dataloader()

    image_names = datamodule.test_data.dataset.get_image_names()

    results = []

    func = partial(process_sample,
                   dataset=datamodule.test_data.dataset,
                   prediction_dir=prediction_dir,
                   threshold=threshold,
                   classes=dataset.classes)

    for sample_results in mapper(
            func,
            zip(dataloader, image_names),
            total=len(image_names),
            num_cpus=4,
    ):
        results += sample_results

    outfile = os.path.join(os.path.abspath(base_dir), "metrics.csv")
    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
