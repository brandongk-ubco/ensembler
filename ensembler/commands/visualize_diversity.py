import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

sns.set_theme()
matplotlib.use('Agg')


def visualize_diversity(in_dir: str):

    configs_file = os.path.join(in_dir, "configs.csv")
    diversity_file = os.path.join(in_dir, "diversity.csv")
    assert os.path.exists(diversity_file)
    assert os.path.exists(configs_file)

    diversity = pd.read_csv(diversity_file)
    configs = pd.read_csv(configs_file)

    best_models = configs[configs["best"] == True]

    diversity["best"] = diversity["left_job_hash"].isin(
        best_models["job_hash"]) & diversity["right_job_hash"].isin(
            best_models["job_hash"])
    diversity.to_csv(os.path.join(in_dir, "diversity.csv"))

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
