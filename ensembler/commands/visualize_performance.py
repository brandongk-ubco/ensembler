import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.perf import RegressionPerf

sns.set(style="whitegrid")
matplotlib.use('Agg')


def plot_miou_by_class(df, base_dir):
    plot = sns.boxplot(x='class',
                       y='value',
                       data=df,
                       color="skyblue",
                       hue="Dataset")

    fig = plot.get_figure()

    for patch in plot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))

    plot.set_yticks(np.linspace(0, 1, num=21))

    plot.set_xlabel('Class')
    plot.set_ylabel('mIoU')
    plot.set_title('mIoU by Class')
    for tick in plot.get_xticklabels():
        tick.set_rotation(90)

    outfile = os.path.join(base_dir, "performance.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def visualize_performance(base_dir: str):

    metrics_file = os.path.join(base_dir, "metrics.csv")

    assert os.path.exists(metrics_file)

    df = pd.read_csv(metrics_file)
    grouped = df.groupby(
        by=["job_hash", "class", "metric", "model_activation", "type"
           ]).mean().reset_index()

    iou_df = grouped[grouped.metric == "iou"].sort_values("value")

    def renaming_fun(x):
        return x.replace("model_", "")

    iou_df = iou_df.rename(columns=renaming_fun)

    iou_df["width_ratio"] = (iou_df["width_ratio"] * 10).astype(int)
    iou_df = iou_df.rename(columns={"type": "Dataset"})

    plot_miou_by_class(iou_df, base_dir)
