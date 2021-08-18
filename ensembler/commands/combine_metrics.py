import glob
import os
import pandas as pd
import yaml
from flatten_dict import flatten
from ensembler.p_tqdm import t_imap as mapper


def process_file(file_path: str) -> pd.DataFrame:
    job_hash = os.path.split(os.path.dirname(file_path))[-1]
    metrics = pd.read_csv(file_path)
    config_file = os.path.join(os.path.dirname(file_path), "config.yaml")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = flatten(config, reducer="underscore")
    for key, val in config.items():
        metrics[key] = val
    metrics["job_hash"] = job_hash

    return metrics


def combine_metrics(in_dir: str):
    in_dir = os.path.abspath(in_dir)
    job_hashes = [
        d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
    ]

    metrics_files = glob.glob(os.path.join(in_dir, "**", "metrics.csv"))
    metric_directories = [
        os.path.split(os.path.dirname(f))[-1] for f in metrics_files
    ]

    missing_metrics = [m for m in job_hashes if m not in metric_directories]

    if missing_metrics:
        raise FileExistsError(
            "Could not find metrics for: {}".format(missing_metrics))

    metrics = []

    for df in mapper(process_file, metrics_files):
        metrics.append(df)

    metrics = pd.concat(metrics)

    outfile = os.path.join(in_dir, "metrics.csv")
    metrics.to_csv(outfile)