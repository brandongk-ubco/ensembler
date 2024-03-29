import glob
import os
import pandas as pd
import yaml
from flatten_dict import flatten
from ensembler.p_tqdm import t_imap as mapper
import re
from functools import partial

from ensembler.Dataset import Dataset
from ensembler.datasets import Datasets


def process_file(file_path: str) -> pd.DataFrame:
    file_dir = os.path.dirname(file_path)
    job_hash = os.path.split(file_dir)[-1]
    metrics = pd.read_csv(file_path)
    config_file = os.path.join(file_dir, "config.yaml")

    models = glob.glob(
        os.path.join(file_dir, "lightning_logs", "**", "checkpoints", "*.ckpt"))

    model_scores = [
        float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', os.path.basename(m))[-1])
        for m in models
    ]

    model_idx = model_scores.index(min(model_scores))

    model = models[model_idx]

    epoch, loss = re.findall(r'[-+]?[0-9]*\.?[0-9]+', os.path.basename(model))
    epoch = int(epoch)
    loss = float(loss)

    with open(os.path.join(file_dir, "predict_time.txt")) as pt:
        predict_time = float(pt.readline())

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = flatten(config, reducer="underscore")
    for key, val in config.items():
        metrics[key] = val

    metrics["job_hash"] = job_hash
    metrics["epoch"] = epoch
    metrics["loss"] = loss
    metrics["predict_time"] = predict_time

    return metrics


def combine_metrics(in_dir: str):

    dataset = Datasets["cityscapes"]
    datamodule = Dataset(dataset=dataset, batch_size=1)
    dataset = Datasets.get(dataset.value)
    dataloader = datamodule.test_dataloader()
    image_names = datamodule.test_data.dataset.get_image_names()

    in_dir = os.path.abspath(in_dir)
    job_hashes = [
        d for d in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, d)) and
        d not in ["ebms", "ensembles", "test", "val"]
    ]

    metrics_files = glob.glob(os.path.join(in_dir, "**", "metrics.csv"))
    metric_directories = [
        os.path.split(os.path.dirname(f))[-1] for f in metrics_files
    ]

    missing_metrics = [m for m in job_hashes if m not in metric_directories]

    if missing_metrics:
        raise FileExistsError(
            "Could not find metrics for: {}".format(missing_metrics))

    combined_metrics = []

    for df in mapper(process_file, metrics_files):
        combined_metrics.append(df)

    combined_metrics = pd.concat(combined_metrics, ignore_index=True)

    def type_mapper(image_name, datamodule):
        if image_name in datamodule.test_data.dataset.test_images:
            return "test"
        if image_name in datamodule.test_data.dataset.val_images:
            return "val"
        import pdb
        pdb.set_trace()

    combined_metrics["type"] = combined_metrics["image"].apply(
        partial(type_mapper, datamodule=datamodule))

    outfile = os.path.join(in_dir, "metrics.csv")
    combined_metrics.to_csv(outfile, index=False)
