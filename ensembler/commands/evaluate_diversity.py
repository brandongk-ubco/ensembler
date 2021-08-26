import os
from functools import lru_cache
import yaml
from flatten_dict import flatten
from functools import partial
from ensembler.p_tqdm import t_imap as outer_mapper, p_imap as loader_mapper, t_imap as hash_mapper
import glob
import numpy as np
import pandas as pd

classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]


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


def load_prediction(image_name, prediction_dir):
    prediction_file = os.path.join(prediction_dir, image_name)
    assert os.path.exists(prediction_file)
    prediction = np.load(prediction_file)["predicted_mask"]
    prediction = prediction.reshape(-1, prediction.shape[-1])
    return prediction[::10, :]


def compare_hash_for_job_and_class(clazz_idx, left, right):
    left_clazz = left[:, clazz_idx]
    right_clazz = right[:, clazz_idx]
    return clazz_idx, np.corrcoef(left_clazz, right_clazz)[0, 1]


def compare_hash_for_job(iteration, job_hashes, config_fetcher, in_dir):

    i, job_hash = iteration

    left_config = {**config_fetcher(job_hash=job_hash)}
    left_config.pop("data_dataset")
    left_config = dict([
        ("left_{}".format(k), v) for k, v in left_config.items()
    ])

    left_predictions_dir = os.path.join(in_dir, job_hash, "predictions")
    left_image_paths = glob.glob(os.path.join(left_predictions_dir, "*.npz"))
    image_names = [os.path.basename(i) for i in left_image_paths]

    left = []
    left_loader = partial(load_prediction, prediction_dir=left_predictions_dir)

    for left_prediction in loader_mapper(left_loader,
                                         image_names,
                                         num_cpus=os.cpu_count() // 2):
        left.append(left_prediction)

    left = np.concatenate(left, axis=0)

    results = []

    for compare_hash in job_hashes[i + 1:]:

        right_predictions_dir = os.path.join(in_dir, compare_hash,
                                             "predictions")

        right_config = {**config_fetcher(job_hash=compare_hash)}
        config = {**left_config, **right_config}

        right = []
        right_loader = partial(load_prediction,
                               prediction_dir=right_predictions_dir)

        for right_prediction in loader_mapper(right_loader,
                                              image_names,
                                              num_cpus=os.cpu_count() // 2):
            right.append(right_prediction)

        right = np.concatenate(right, axis=0)

        comparator = partial(compare_hash_for_job_and_class,
                             left=left,
                             right=right)
        for clazz_idx, correlation in hash_mapper(comparator,
                                                  range(left.shape[1]),
                                                  num_cpus=os.cpu_count() // 2):
            clazz = classes[clazz_idx]
            result = {**config}
            result["class"] = clazz
            result["left_job_hash"] = job_hash
            result["right_job_hash"] = compare_hash
            result["correlation"] = correlation
            results.append(result)

    return results


def evaluate_diversity(in_dir: str):
    in_dir = os.path.abspath(in_dir)

    config_fetcher = partial(get_config, base_dir=in_dir)
    results = []
    outfile = os.path.join(in_dir, "diversity2.csv")

    results = None
    if os.path.exists(outfile):
        results = pd.read_csv(outfile)

    job_hashes = sorted([
        d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
    ])

    hash_comparator = partial(compare_hash_for_job,
                              job_hashes=job_hashes,
                              config_fetcher=config_fetcher,
                              in_dir=in_dir)

    results = []

    for result in outer_mapper(hash_comparator,
                               enumerate(job_hashes[:-1]),
                               total=len(job_hashes) - 1):
        results += result

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
