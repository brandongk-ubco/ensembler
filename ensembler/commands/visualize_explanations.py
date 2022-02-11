import os
import pandas as pd
from typing import List
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
matplotlib.use('Agg')


def visualize_explanations(base_dir: str):

    ebm_dir = os.path.join(base_dir, "ebms")
    iou_dir = os.path.join(ebm_dir, "IoU")
    correlation_dir = os.path.join(ebm_dir, "correlation")
    agreement_dir = os.path.join(ebm_dir, "agreement")
    iou_scores_df = pd.read_csv(os.path.join(iou_dir, "scores.csv"))
    iou_details_df = pd.read_csv(os.path.join(iou_dir, "details.csv"))

    correlation_scores_df = pd.read_csv(
        os.path.join(correlation_dir, "scores.csv"))
    correlation_details_df = pd.read_csv(
        os.path.join(correlation_dir, "details.csv"))

    agreement_scores_df = pd.read_csv(os.path.join(agreement_dir, "scores.csv"))
    agreement_details_df = pd.read_csv(
        os.path.join(agreement_dir, "details.csv"))

    scores_df = pd.merge(iou_scores_df,
                         agreement_scores_df,
                         on="class",
                         suffixes=("_iou", "_agreement")).merge(
                             correlation_scores_df, on="class").rename(columns={
                                 "score": "Correlation",
                                 "score_agreement": "Agreement",
                                 "score_iou": "IoU"
                             }).sort_values("IoU", ascending=False)

    scores_df.loc['mean'] = scores_df.iloc[1:].mean()
    scores_df.at['mean', 'class'] = "mean"
    scores_df.loc['median'] = scores_df.iloc[1:-1].median()
    scores_df.at['median', 'class'] = "median"
    scores_df.loc['std'] = scores_df.iloc[1:-2].std()
    scores_df.at['std', 'class'] = "std"

    scores_df.to_latex(os.path.join(ebm_dir, "explanation_quality.tex"),
                       float_format="%.2f",
                       index=False)
    # scores_df = pd.melt(scores_df,
    #                     id_vars=['class'],
    #                     value_vars=["Agreement", "Correlation", "IoU"
    #                                ]).rename(columns={"variable": "category"})

    # plot = sns.boxplot(data=scores_df, x="class", y="value", hue="category")

    # fig = plot.get_figure()
    # ax = fig.get_axes()[0]

    # ax.set_xlabel("Class")
    # ax.set_ylabel("Prediction Quality $(R^2)$")
    # ax.set_title("Prediction Quality by Class and Category")

    # outfile = os.path.join(ebm_dir, "prediction_quality.png")
    # fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.close()
