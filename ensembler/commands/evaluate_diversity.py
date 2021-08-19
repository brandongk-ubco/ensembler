import os
from functools import lru_cache
import yaml
from flatten_dict import flatten
from functools import partial
from ensembler.p_tqdm import t_imap as mapper
import glob
import numpy as np


@lru_cache(maxsize=None)
def get_config(base_dir, job_hash):
    file_dir = os.path.join(base_dir, job_hash)
    config_file = os.path.join(file_dir, "config.yaml")

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = flatten(config, reducer="underscore")
    return config


def load_prediction(prediction_file):
    assert os.path.exists(prediction_file)
    mask = np.load(prediction_file)["predicted_mask"].astype(np.float32)
    import pdb
    pdb.set_trace()
    return mask / 255


def compare_hashes(hashes, config_fetcher, in_dir):
    hash, compare_hash = hashes
    left_predictions_dir = os.path.join(in_dir, hash, "predictions")
    right_predictions_dir = os.path.join(in_dir, compare_hash, "predictions")
    left_config = config_fetcher(job_hash=hash)
    right_config = config_fetcher(job_hash=compare_hash)

    assert left_config["data_dataset"] == right_config["data_dataset"]
    left_config.pop("data_dataset")
    right_config.pop("data_dataset")

    left_config = dict([
        ("left_{}".format(k), v) for k, v in left_config.items()
    ])

    right_config = dict([
        ("right_{}".format(k), v) for k, v in left_config.items()
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

    for image_name in left_images:
        left_prediction = load_prediction(
            os.path.join(left_predictions_dir, image_name))
        right_prediction = load_prediction(
            os.path.join(right_predictions_dir, image_name))
        import pdb
        pdb.set_trace()


def evaluate_diversity(in_dir: str):
    in_dir = os.path.abspath(in_dir)

    config_fetcher = partial(get_config, base_dir=in_dir)
    job_hashes = sorted([
        d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
    ])

    results = []

    comparisons = []

    for i, hash in enumerate(job_hashes):
        for compare_hash in job_hashes[i + 1:]:
            comparisons.append((hash, compare_hash))

    comparator = partial(compare_hashes,
                         config_fetcher=config_fetcher,
                         in_dir=in_dir)

    for result in mapper(comparator, comparisons):
        results.append(result)
