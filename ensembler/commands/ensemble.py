from ensembler.methods import EnsembleMethods
from argh import arg
import pandas as pd
import os
import yaml
from flatten_dict import flatten
from functools import lru_cache, partial
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')


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


@arg('method_name', choices=EnsembleMethods.choices())
def ensemble(in_dir: str, method_name: str, p_thresh: float = 0.05):
    assert method_name is not None

    job_hashes = sorted([
        d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
    ])

    config_fetcher = partial(get_config, base_dir=in_dir)

    configs_list = []

    for job_hash in job_hashes:
        config = config_fetcher(job_hash=job_hash)
        configs_list.append(config)

    configs = pd.DataFrame(configs_list)

    method = EnsembleMethods.get(method_name)

    metrics_file = os.path.join(in_dir, "metrics.csv")
    diversity_file = os.path.join(in_dir, "diversity.csv")
    tukey_file = os.path.join(in_dir, "tukey.csv")
    assert os.path.exists(metrics_file)
    assert os.path.exists(diversity_file)
    assert os.path.exists(tukey_file)

    metrics = pd.read_csv(metrics_file)
    diversity = pd.read_csv(diversity_file)
    diversity = diversity[[
        "left_job_hash", "right_job_hash", "agreement",
        "disagreement_correlation", "class"
    ]]
    tukey = pd.read_csv(tukey_file).rename(columns={"Unnamed: 0": "job_hashes"})
    tukey["better_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[0], axis=1)
    tukey["worse_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[1], axis=1)
    mIoU = metrics[metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "mIoU"
    }).sort_values(by="mIoU", ascending=False).reset_index()

    statistically_same = {}

    for idx, row in tukey.iterrows():
        better = row["better_job_hash"]
        worse = row["worse_job_hash"]
        are_same = row["p_adj"] >= p_thresh

        if better not in statistically_same:
            statistically_same[better] = []
        if worse not in statistically_same:
            statistically_same[worse] = []

        if are_same:
            statistically_same[better].append(worse)
            statistically_same[worse].append(better)

    best_models = statistically_same["bfd33d38ce9c4d84b912f1ee0b7961aa"]
    best_models.append("bfd33d38ce9c4d84b912f1ee0b7961aa")

    diversity["best"] = diversity["left_job_hash"].isin(
        best_models) & diversity["right_job_hash"].isin(best_models)

    best_configs = configs[configs["job_hash"].isin(best_models)]
    best_diversity = diversity[
        diversity["left_job_hash"].isin(best_models) &
        diversity["right_job_hash"].isin(best_models)].sort_values(
            by=["disagreement_correlation", "agreement"],
            ascending=[True, True])

    sns.set(rc={'figure.figsize': (11, 11)})

    classes = diversity["class"].unique().tolist()
    for clazz in classes:
        class_data = diversity[diversity["class"] == clazz].sort_values(
            "best", ascending=True)
        plot = sns.scatterplot(data=class_data,
                               x='disagreement_correlation',
                               y='agreement',
                               hue="best",
                               legend=False)
        fig = plot.get_figure()

        plt.title(clazz)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        outfile = os.path.join(in_dir, "agreement_{}.png".format(clazz))
        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
