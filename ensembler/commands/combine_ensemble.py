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


@arg('dataset', choices=Datasets.choices())
@arg('job_hashes', nargs='+')
def combine_ensemble(dataset: Datasets, base_dir: str, job_hashes: List[str]):

    metrics_file = os.path.join(base_dir, "metrics.csv")

    print("Loading {}".format(metrics_file))
    metrics = pd.read_csv(metrics_file)
    metrics = metrics[["image", "class", "threshold", "metric", "value"]]
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
        results.append(metrics)
    job_hashes.sort()
    ensemble_hash = hashlib.md5("".join(job_hashes).encode()).hexdigest()
    outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                           "combined_metrics.csv")
    print("Writing {}".format(outfile))
    df = pd.concat(results)
    df.to_csv(outfile, index=False)
