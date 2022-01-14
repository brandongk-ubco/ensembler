import os
import pandas as pd
import yaml
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')
sns.set_theme()


def statistical_similarity(in_dir: str, p_thresh: float = 0.05):
    tukey_file = os.path.join(in_dir, "tukey.csv")
    metrics_file = os.path.join(in_dir, "metrics.csv")
    assert os.path.exists(tukey_file)
    assert os.path.exists(metrics_file)

    metrics = pd.read_csv(metrics_file)

    tukey = pd.read_csv(tukey_file).rename(columns={"Unnamed: 0": "job_hashes"})
    tukey["better_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[0], axis=1)
    tukey["worse_job_hash"] = tukey.apply(
        lambda row: row["job_hashes"].split("-")[1], axis=1)

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

    with open(os.path.join(in_dir, "statistically_same.yaml"), "w") as yamlfile:
        yaml.dump(statistically_same, yamlfile, indent=4)

    mIoU = metrics[metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "mIoU"
    }).sort_values(by="mIoU", ascending=False).reset_index()

    ranked_mIoU = mIoU.reset_index().rename(columns={"index": "rank"})
    ranked_mIoU["rank"] = ranked_mIoU["rank"] + 1
    ranked_mIoU["statistically_same_count"] = ranked_mIoU["job_hash"].apply(
        lambda job_hash: len(statistically_same[job_hash]))

    plot = sns.barplot(data=ranked_mIoU,
                       x='rank',
                       y='statistically_same_count',
                       color='black')
    fig = plot.get_figure()

    plt.xticks([0, len(ranked_mIoU) - 1])
    plt.tight_layout()

    outfile = os.path.join(in_dir, "statistical_same_count.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
