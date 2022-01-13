from ensembler.Dataset import Dataset
from ensembler.datasets import Datasets
from argh import arg
import os
import numpy as np
from ensembler.utils import classwise
from ensembler.p_tqdm import p_uimap as mapper
from functools import partial
import pandas as pd
from typing import List
import hashlib
import json
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


def visualize_ensemble(base_dir: str, ensemble_hash: List[str]):

    metrics_file = os.path.join(base_dir, "ensembles", ensemble_hash,
                                "combined_metrics.csv")
    df = pd.read_csv(metrics_file)
    df = df[df["metric"] == "iou"]
    df = df.rename(columns={"metric": "IoU"})

    for clazz in df["class"].unique():
        class_data = df[df["class"] == clazz]

        plot = sns.boxplot(data=class_data, x='ensemble_size', y='value')
        fig = plot.get_figure()

        outfile = os.path.join(base_dir, "ensembles", ensemble_hash,
                               "{}.png".format(clazz))
        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
