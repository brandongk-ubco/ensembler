import yaml
from flatten_dict import flatten
from functools import lru_cache, partial
import os
import pandas as pd


@lru_cache(maxsize=None)
def get_config(base_dir, job_hash):
    file_dir = os.path.join(base_dir, job_hash)
    config_file = os.path.join(file_dir, "config.yaml")

    assert os.path.exists(config_file)

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


def activation_mapper(activation: str):
    if activation == "leaky_relu":
        return "Leaky ReLU"
    if activation == "swish":
        return "Swish"
    if activation == "relu":
        return "ReLU"
    if activation == "piecewise_linear":
        return "Piecewise Linear"

    raise ValueError("Unknown activation: {}".format(activation))


def combine_configs(in_dir: str):
    job_hashes = sorted([
        d for d in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, d)) and d != "ensembles"
    ])

    config_fetcher = partial(get_config, base_dir=in_dir)

    configs_list = []

    for job_hash in job_hashes:
        config = config_fetcher(job_hash=job_hash)
        configs_list.append(config)

    configs = pd.DataFrame(configs_list)

    metrics_file = os.path.join(in_dir, "metrics.csv")
    similarity_file = os.path.join(in_dir, "statistically_same.yaml")

    assert os.path.exists(similarity_file)
    assert os.path.exists(metrics_file)

    with open(similarity_file, "r") as similarity_yaml:
        statistically_same = yaml.safe_load(similarity_yaml)

    metrics = pd.read_csv(metrics_file)

    mIoU = metrics[metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "mIoU"
    }).sort_values(by="mIoU", ascending=False).reset_index()

    ranked_mIoU = mIoU.reset_index().rename(columns={"index": "rank"})
    ranked_mIoU["rank"] = ranked_mIoU["rank"] + 1

    configs = configs.set_index('job_hash').join(
        ranked_mIoU.set_index('job_hash')).sort_values(
            "rank", ascending=True).reset_index()

    best_hash = configs.iloc[0]["job_hash"]

    best_models = statistically_same[best_hash]
    best_models.append(best_hash)

    configs["best"] = configs["job_hash"].isin(best_models)

    best_configs = configs[configs["best"] == True]
    best_configs = best_configs[[
        "model_activation", "model_depth", "model_residual_units",
        "model_width", "model_width_ratio", "mIoU"
    ]]
    best_configs = best_configs.rename(
        columns={
            "model_activation": "Activation",
            "model_depth": "Depth",
            "model_residual_units": "Residual Units",
            "model_width": "Width",
            "model_width_ratio": "Width Ratio",
        })

    best_configs["Activation"] = best_configs["Activation"].apply(
        lambda activation: activation_mapper(activation))
    best_configs.to_latex(os.path.join(in_dir, "best_models.tex"),
                          index=False,
                          float_format="%.3f")

    configs.to_csv(os.path.join(in_dir, "configs.csv"), index=False)
