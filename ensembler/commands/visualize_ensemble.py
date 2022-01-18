import os
import pandas as pd
from typing import List
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
matplotlib.use('Agg')


def visualize_ensemble(base_dir: str, ensemble_hash: List[str]):

    metrics_file = os.path.join(base_dir, "ensembles", ensemble_hash,
                                "combined_metrics.csv")
    df = pd.read_csv(metrics_file)
    df = df[df["metric"] == "iou"]
    df = df.rename(columns={"metric": "IoU"})

    for clazz in df["class"].unique():
        print(clazz)
        class_data = df[df["class"] == clazz]

        plot = sns.boxplot(data=class_data,
                           x='ensemble_size',
                           y='value',
                           color='lightblue')
        fig = plot.get_figure()
        ax = fig.get_axes()[0]

        ax.set_xlabel("Ensemble Size")
        ax.set_ylabel("mIoU")

        ax.set_title("mIoU by Ensemble Size - {}".format(clazz.title()))

        plot.set_yticks(np.linspace(0, 1, num=11))

        outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                               "{}.png".format(clazz))
        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    print("Loading Improvement")
    improvement_file = os.path.join(base_dir, "ensembles", ensemble_hash,
                                    "improvement.csv")

    improvement = pd.read_csv(improvement_file)
    improvement = improvement.dropna()

    print("Plotting Improvement")

    plot = sns.boxplot(data=improvement,
                       x='ensemble_size',
                       y='improvement',
                       color='lightblue')
    fig = plot.get_figure()
    ax = fig.get_axes()[0]

    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("mIoU Improvement over Previous")

    ax.set_title("Improvement by Ensemble Size")

    # plot.set_yticks(np.linspace(0, 1, num=11))

    print("Saving Improvement plot")

    outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                           "improvement.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    mIoU = improvement.groupby(by="ensemble_size").mean().sort_values(
        by="ensemble_size", ascending=False).reset_index()
    plot = sns.scatterplot(data=mIoU,
                           x='ensemble_size',
                           y='improvement',
                           legend=False)
    fig = plot.get_figure()

    outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                           "mean_improvement.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
