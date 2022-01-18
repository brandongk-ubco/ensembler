from argh import arg
import os
from functools import partial
import pandas as pd
from typing import List
import hashlib

from functools import partial
from tqdm import tqdm

tqdm.pandas()


def calculate_improvement(df, current_row):

    ensemble_size = current_row["ensemble_size"]
    image = current_row["image"]
    clazz = current_row["class"]
    if ensemble_size == 1:
        return None

    assert ensemble_size > 1
    assert ensemble_size <= df["ensemble_size"].max()

    previous_row = df[(df["image"] == image) & (df["class"] == clazz) &
                      (df["ensemble_size"] == ensemble_size - 1)]

    assert len(previous_row) == 1

    previous_value = previous_row["value"]
    value = current_row["value"]

    improvement = (value - previous_value)

    return improvement.iloc[0]


@arg('job_hashes', nargs='+')
def combine_ensemble(base_dir: str, job_hashes: List[str]):

    metrics_file = os.path.join(base_dir, "metrics.csv")

    print("Loading {}".format(metrics_file))
    metrics = pd.read_csv(metrics_file)
    metrics = metrics.rename(columns={"job_hash": "ensemble_hash"})
    metrics = metrics[metrics["ensemble_hash"] == job_hashes[0]]
    metrics = metrics[[
        "image", "class", "threshold", "metric", "value", "ensemble_hash"
    ]]
    metrics["ensemble_size"] = 1

    results = [metrics]
    for idx in range(1, len(job_hashes)):
        hashes = job_hashes[:idx + 1]
        hashes.sort()
        ensemble_hash = hashlib.md5("".join(hashes).encode()).hexdigest()
        metrics_file = os.path.join(base_dir, "ensembles", ensemble_hash,
                                    "metrics.csv")
        print("Loading {}".format(metrics_file))
        metrics = pd.read_csv(metrics_file)
        metrics["ensemble_size"] = len(hashes)
        metrics["ensemble_hash"] = ensemble_hash
        results.append(metrics)
    job_hashes.sort()
    ensemble_hash = hashlib.md5("".join(job_hashes).encode()).hexdigest()
    outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                           "combined_metrics.csv")

    print("Writing {}".format(outfile))
    df = pd.concat(results)
    df.to_csv(outfile, index=False)

    iou = df[df["metric"] == "iou"].copy(deep=True)

    improvement_caluculator = partial(calculate_improvement, df=iou)

    iou["improvement"] = iou.progress_apply(
        lambda row: improvement_caluculator(current_row=row), axis=1)

    outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                           "improvement.csv")
    iou.to_csv(outfile, index=False)
