import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

sns.set_theme()
matplotlib.use('Agg')


def scatterplots(configs, diversity, in_dir: str):
    best_models = configs[configs["best"] == True]

    diversity["best"] = diversity["left_job_hash"].isin(
        best_models["job_hash"]) & diversity["right_job_hash"].isin(
            best_models["job_hash"])

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

        outfile = os.path.join(in_dir,
                               "pairwise_diversity_{}.png".format(clazz))
        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


def build_df(in_dir: str):

    configs_file = os.path.join(in_dir, "configs.csv")
    diversity_file = os.path.join(in_dir, "diversity.csv")
    assert os.path.exists(diversity_file)
    assert os.path.exists(configs_file)

    diversity = pd.read_csv(diversity_file)
    configs = pd.read_csv(configs_file)

    left_configs = configs.copy(deep=True)
    left_configs = left_configs.add_prefix("left_")
    right_configs = configs.copy(deep=True)
    right_configs = right_configs.add_prefix("right_")
    diversity = diversity.loc[:, ~diversity.columns.str.contains('^Unnamed')]
    diversity = diversity[[
        "left_job_hash", "right_job_hash", "correlation",
        "disagreement_correlation", "class"
    ]]

    diversity = diversity.merge(left_configs, on="left_job_hash", how="inner")
    diversity = diversity.merge(right_configs, on="right_job_hash", how="inner")

    def activation(row):
        activation = [
            row["left_model_activation"], row["right_model_activation"]
        ]
        activation.sort()
        activation = [str(a) for a in activation]
        return "-".join(activation)

    def residual_units(row):
        residual_units = [
            row["left_model_residual_units"], row["right_model_residual_units"]
        ]
        residual_units.sort()
        residual_units = [str(a) for a in residual_units]
        return "-".join(residual_units)

    def width(row):
        width = [row["left_model_width"], row["right_model_width"]]
        width.sort()
        width = [str(a) for a in width]
        return "-".join(width)

    def width_ratio(row):
        width_ratio = [
            row["left_model_width_ratio"], row["right_model_width_ratio"]
        ]
        width_ratio.sort()
        width_ratio = [str(a) for a in width_ratio]
        return "-".join(width_ratio)

    def depth(row):
        depth = [row["left_model_depth"], row["right_model_depth"]]
        depth.sort()
        depth = [str(a) for a in depth]
        return "-".join(depth)

    def performance_diff(row):
        return abs(row["right_test_mIoU"] - row["left_test_mIoU"])

    print("Combining Model Specs")
    diversity["performance_diff"] = diversity.apply(performance_diff, axis=1)
    diversity["activation"] = diversity.apply(activation, axis=1)
    diversity["residual_units"] = diversity.apply(residual_units, axis=1)
    diversity["width"] = diversity.apply(width, axis=1)
    diversity["width_ratio"] = diversity.apply(width_ratio, axis=1)
    diversity["depth"] = diversity.apply(depth, axis=1)

    return configs, diversity


def visualize_diversity(in_dir: str):

    configs, diversity = build_df(in_dir)
    scatterplots(configs, diversity, in_dir)
