import os
from functools import lru_cache
import yaml
from flatten_dict import flatten
from functools import partial
from ensembler.p_tqdm import t_imap as outer_mapper, p_imap as inner_mapper
import glob
import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def get_config(base_dir, job_hash):
    file_dir = os.path.join(base_dir, job_hash)
    config_file = os.path.join(file_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise AttributeError("Couldn't find config file {}".format(config_file))

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = flatten(config, reducer="underscore")
    config["job_hash"] = job_hash

    if "data_dataset" not in config:
        raise AttributeError(
            "Expected data_dataset to be in {}".format(config_file))
    return config


def load_prediction(prediction_file):
    assert os.path.exists(prediction_file)
    mask = np.load(prediction_file)["predicted_mask"].astype(np.float32)
    return mask / 255


def compare_predictions(image_name, config, left_predictions_dir,
                        right_predictions_dir):

    left_prediction = load_prediction(
        os.path.join(left_predictions_dir, image_name))
    right_prediction = load_prediction(
        os.path.join(right_predictions_dir, image_name))

    difference = np.abs(left_prediction - right_prediction).mean()

    return difference


def allbut(levels, names):
    names = set(names)
    return [item for item in levels if item not in names]


def compare_hashes(hashes, config_fetcher, in_dir):
    hash, compare_hash = hashes
    left_predictions_dir = os.path.join(in_dir, hash, "predictions")
    right_predictions_dir = os.path.join(in_dir, compare_hash, "predictions")
    left_config = {**config_fetcher(job_hash=hash)}
    right_config = {**config_fetcher(job_hash=compare_hash)}

    if "data_dataset" not in left_config:
        raise AttributeError("Couldn't find data_dataset in {}".format(hash))

    if "data_dataset" not in right_config:
        raise AttributeError(
            "Couldn't find data_dataset in {}".format(compare_hash))

    assert left_config["data_dataset"] == right_config["data_dataset"]
    left_config.pop("data_dataset")
    right_config.pop("data_dataset")

    left_config = dict([
        ("left_{}".format(k), v) for k, v in left_config.items()
    ])

    right_config = dict([
        ("right_{}".format(k), v) for k, v in right_config.items()
    ])

    config = {**left_config, **right_config}

    left_images = [
        os.path.basename(i)
        for i in glob.glob(os.path.join(left_predictions_dir, "*.npz"))
    ]
    right_images = [
        os.path.basename(i)
        for i in glob.glob(os.path.join(right_predictions_dir, "*.npz"))
    ]

    missing_images = list(set(left_images).difference(right_images)) + list(
        set(right_images).difference(left_images))

    if missing_images:
        raise AssertionError(
            "Expected all images to be in both folders, but some are not: {}".
            format(missing_images))

    result_comparator = partial(compare_predictions,
                                config=config,
                                left_predictions_dir=left_predictions_dir,
                                right_predictions_dir=right_predictions_dir)

    differences = []
    for result in inner_mapper(result_comparator,
                               left_images,
                               num_cpus=os.cpu_count() // 2):
        differences.append(result)

    results = config
    results["difference"] = np.mean(differences)

    return results


def evaluate_diversity(in_dir: str):
    in_dir = os.path.abspath(in_dir)

    config_fetcher = partial(get_config, base_dir=in_dir)
    results = []
    comparisons = []
    outfile = os.path.join(in_dir, "diversity.csv")

    results = None
    if os.path.exists(outfile):
        results = pd.read_csv(outfile)

    job_hashes = sorted([
        d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
    ])

    for i, hash in enumerate(job_hashes):
        for compare_hash in job_hashes[i + 1:]:
            comparisons.append((hash, compare_hash))

    hash_comparator = partial(compare_hashes,
                              config_fetcher=config_fetcher,
                              in_dir=in_dir)

    results = []

    for result in outer_mapper(hash_comparator,
                               comparisons,
                               num_cpus=os.cpu_count() // 2):
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
