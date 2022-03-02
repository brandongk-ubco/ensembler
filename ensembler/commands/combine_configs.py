import yaml
from flatten_dict import flatten
from functools import lru_cache, partial
import os
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

sns.set(style="whitegrid")
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = [11, 5]


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
        if os.path.isdir(os.path.join(in_dir, d)) and
        d not in ["ensembles", "test", "val", "ebms"]
    ])

    config_fetcher = partial(get_config, base_dir=in_dir)

    configs_list = []

    for job_hash in job_hashes:
        config = config_fetcher(job_hash=job_hash)
        configs_list.append(config)

    configs = pd.DataFrame(configs_list)

    metrics_file = os.path.join(in_dir, "metrics.csv")
    test_similarity_file = os.path.join(in_dir, "test_statistically_same.yaml")
    val_similarity_file = os.path.join(in_dir, "val_statistically_same.yaml")
    tukey_file = os.path.join(in_dir, "test_tukey.csv")

    assert os.path.exists(tukey_file)
    assert os.path.exists(test_similarity_file)
    assert os.path.exists(val_similarity_file)
    assert os.path.exists(metrics_file)

    tukey = pd.read_csv(tukey_file).rename(columns={"Unnamed: 0": "job_hashes"})
    tukey["better_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[0], axis=1)
    tukey["worse_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[1], axis=1)

    with open(test_similarity_file, "r") as similarity_yaml:
        test_statistically_same = yaml.safe_load(similarity_yaml)

    with open(val_similarity_file, "r") as similarity_yaml:
        val_statistically_same = yaml.safe_load(similarity_yaml)

    metrics = pd.read_csv(metrics_file)

    test_metrics = metrics[metrics["type"] == "test"]
    val_metrics = metrics[metrics["type"] == "val"]

    test_mIoU = test_metrics[test_metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "test_mIoU"
    }).sort_values(by="test_mIoU", ascending=False).reset_index()
    test_ranked_mIoU = test_mIoU.reset_index().rename(
        columns={"index": "test_rank"})
    test_ranked_mIoU["test_rank"] = test_ranked_mIoU["test_rank"] + 1

    val_mIoU = val_metrics[val_metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "val_mIoU"
    }).sort_values(by="val_mIoU", ascending=False).reset_index()
    val_ranked_mIoU = val_mIoU.reset_index().rename(
        columns={"index": "val_rank"})
    val_ranked_mIoU["val_rank"] = val_ranked_mIoU["val_rank"] + 1

    configs = configs.set_index('job_hash').join(
        test_ranked_mIoU.set_index('job_hash')).join(
            val_ranked_mIoU.set_index('job_hash')).sort_values(
                "test_rank", ascending=True).reset_index()

    best_hash = configs.iloc[0]["job_hash"]

    best_models = test_statistically_same[best_hash]

    with open(os.path.join(in_dir, "best_info.yaml"), "w") as yamlfile:
        yaml.dump(
            {
                "best": best_hash,
                "similar": test_statistically_same[best_hash]
            }, yamlfile)

    best_models.append(best_hash)

    configs["best"] = configs["job_hash"].isin([best_hash] + best_models)

    best_configs = configs[configs["best"] == True]

    best_configs = best_configs.sort_values(by="test_rank", ascending=True)

    best_configs = best_configs[[
        "test_rank", "val_rank", "model_activation", "model_depth",
        "model_residual_units", "model_width", "model_width_ratio", "test_mIoU"
    ]]
    best_configs = best_configs.rename(
        columns={
            "test_rank": "Test Rank",
            "val_rank": "Val Rank",
            "model_activation": "Activation",
            "model_depth": "Depth",
            "model_residual_units": "Residual Units",
            "model_width": "Width",
            "model_width_ratio": "Width Ratio",
            "test_mIoU": "mIoU",
        })

    best_configs["Activation"] = best_configs["Activation"].apply(
        lambda activation: activation_mapper(activation))

    best_configs.to_latex(os.path.join(in_dir, "best_models.tex"),
                          index=False,
                          float_format="%.3f")

    configs.to_csv(os.path.join(in_dir, "configs.csv"), index=False)

    test_performance = test_metrics[test_metrics["metric"] == "iou"]

    best_test_metrics = test_performance[test_performance["job_hash"].isin(
        best_models)]

    best_test_metrics = pd.merge(left=best_test_metrics,
                                 right=test_ranked_mIoU,
                                 how="inner",
                                 on="job_hash")
    best_test_metrics = best_test_metrics.sort_values(by="test_rank",
                                                      ascending=True)

    plot = sns.boxplot(x='test_rank',
                       y='value',
                       data=best_test_metrics,
                       color="skyblue")

    fig = plot.get_figure()

    for patch in plot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))

    plot.set_yticks(np.linspace(0, 1, num=21))

    plot.set_xlabel('Rank')
    plot.set_ylabel('IoU')
    plot.set_title('IoU by Rank')

    outfile = os.path.join(in_dir, "best_performance.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    best_tukeys = tukey[tukey["better_job_hash"].isin(best_models) &
                        tukey["worse_job_hash"].isin(best_models)]

    best_tukeys = pd.merge(
        left=best_tukeys.rename(columns={"better_job_hash": "job_hash"}),
        right=test_ranked_mIoU,
        on="job_hash",
        how="inner").rename(columns={"test_rank": "better"})[[
            "diff", "lwr", "upr", "p_adj", "worse_job_hash", "better"
        ]]

    best_tukeys = pd.merge(
        left=best_tukeys.rename(columns={"worse_job_hash": "job_hash"}),
        right=test_ranked_mIoU,
        on="job_hash",
        how="inner").rename(columns={"test_rank": "worse"})[[
            "diff", "lwr", "upr", "p_adj", "better", "worse"
        ]]

    best_tukeys = best_tukeys.set_index(["better", "worse"]).sort_index()

    # best_tukeys["pair"] = best_tukeys[["better", "worse"
    #                                   ]].agg('{0[better]}-{0[worse]}'.format,
    #                                          axis=1)

    best_tukeys.to_latex(os.path.join(in_dir, "best_tukeys.tex"),
                         index=True,
                         float_format="%.3f")
