import os
import pandas as pd
from typing import List
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
matplotlib.use('Agg')


def barplot(df, directory, prefix):

    for dimension in df["dimension"].unique():
        dimension_df = df[df["dimension"] == dimension]
        types = len(dimension_df["type"].unique())
        if dimension.lower() == "importance":
            dimension = "Overall"
        if dimension == "Performance Diff":
            continue

        order_df = dimension_df.copy(deep=True)
        order_df["score"] = order_df["score"].abs()
        order = dimension_df.groupby("name").max().sort_values(
            "score", ascending=False).index.tolist()
        plot = sns.barplot(data=dimension_df,
                           x="name",
                           y="score",
                           color="skyblue",
                           hue="type",
                           order=order)

        plt.legend(bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=types,
                   frameon=False)

        fig = plot.get_figure()
        ax = fig.get_axes()[0]

        ax.set_xlabel("", fontsize=1)
        ax.set_ylabel("", fontsize=1)
        ax.set_title("")
        title = f"Overall {dimension} Importance on {prefix}"
        if dimension == "Overall":
            title = f"Overall Importance on {prefix}"
        fig.suptitle(title, y=1.02, fontsize=14)
        ax.tick_params(axis="x", labelrotation=90)

        ax.yaxis.grid(False)
        plt.axhline(0, alpha=0.5)
        if types > 1:
            ax.xaxis.grid(True, linestyle='dashed', alpha=0.2)
            ax.set_axisbelow(False)

        outfile = os.path.join(directory, f"{title}.png")

        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


def boxplot(df, directory, prefix):

    for dimension in df["dimension"].unique():
        dimension_df = df[df["dimension"] == dimension]
        types = len(dimension_df["type"].unique())
        if dimension.lower() == "importance":
            dimension = "Combined"
        if dimension == "Performance Diff":
            continue

        order_df = dimension_df.copy(deep=True)
        order_df["score"] = order_df["score"].abs()
        order = dimension_df.groupby("name").max().sort_values(
            "score", ascending=False).index.tolist()
        plot = sns.boxplot(data=dimension_df,
                           x="name",
                           y="score",
                           color="skyblue",
                           hue="type",
                           order=order)

        plt.legend(bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=types,
                   frameon=False)

        fig = plot.get_figure()
        ax = fig.get_axes()[0]

        ax.set_xlabel("", fontsize=1)
        ax.set_ylabel("", fontsize=1)
        ax.set_title("")
        title = f"Combined {dimension} Importance on {prefix}"
        if dimension == "Combined":
            title = f"Combined Importance on {prefix}"
        fig.suptitle(title, y=1.02, fontsize=14)
        ax.tick_params(axis="x", labelrotation=90)

        ax.yaxis.grid(False)
        plt.axhline(0, alpha=0.5)
        if types > 1:
            ax.xaxis.grid(True, linestyle='dashed', alpha=0.2)
            ax.set_axisbelow(False)

        outfile = os.path.join(directory, f"{title}.png")

        fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


def overall_scores(base_dir, iou_scores_df, agreement_scores_df,
                   correlation_scores_df):
    scores_df = pd.merge(iou_scores_df,
                         agreement_scores_df,
                         on="class",
                         suffixes=("_iou", "_agreement")).merge(
                             correlation_scores_df, on="class").rename(columns={
                                 "score": "Correlation",
                                 "score_agreement": "Agreement",
                                 "score_iou": "IoU"
                             }).sort_values("IoU", ascending=False)

    scores_boxplot_df = scores_df.copy(deep=True).iloc[1:]

    scores_boxplot_df = pd.melt(
        scores_boxplot_df,
        id_vars=['class'],
        value_vars=["Agreement", "Correlation",
                    "IoU"]).rename(columns={"variable": "category"})

    plot = sns.boxplot(data=scores_boxplot_df,
                       x="category",
                       y="value",
                       color="skyblue",
                       order=["IoU", "Agreement", "Correlation"])

    fig = plot.get_figure()
    ax = fig.get_axes()[0]

    ax.set_xlabel("")
    ax.set_ylabel("Explanation Quality $(R^2)$")
    ax.set_title("Explanation Quality by Category")

    outfile = os.path.join(base_dir, "explanation_quality.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    scores_df.loc['mean'] = scores_df.iloc[1:].mean()
    scores_df.at['mean', 'class'] = "mean"
    scores_df.loc['median'] = scores_df.iloc[1:-1].median()
    scores_df.at['median', 'class'] = "median"
    scores_df.loc['std'] = scores_df.iloc[1:-2].std()
    scores_df.at['std', 'class'] = "std"

    scores_df.to_latex(os.path.join(base_dir, "explanation_quality.tex"),
                       float_format="%.2f",
                       index=False)


def overall_details(base_dir, iou_details_df, agreement_details_df,
                    correlation_details_df):
    iou_overall_df = iou_details_df[iou_details_df["class"] == "overall"].copy(
        deep=True)
    agreement_overall_df = agreement_details_df[agreement_details_df["class"] ==
                                                "overall"].copy(deep=True)
    correlation_overall_df = correlation_details_df[
        correlation_details_df["class"] == "overall"].copy(deep=True)

    iou_overall_df["type"] = "iou"
    agreement_overall_df["type"] = "agreement"
    correlation_overall_df["type"] = "correlation"

    iou_overall_df = iou_overall_df[["name", "score", "type", "dimension"]]
    agreement_overall_df = agreement_overall_df[[
        "name", "score", "type", "dimension"
    ]]
    correlation_overall_df = correlation_overall_df[[
        "name", "score", "type", "dimension"
    ]]

    overall_df = pd.concat(
        [iou_overall_df, agreement_overall_df,
         correlation_overall_df]).copy(deep=True)

    def name_mapping(name):
        return name.replace("_", " ").title()

    def type_mapping(type_txt):
        if type_txt == "iou":
            return "IoU"
        if type_txt == "correlation":
            return "Correlation"
        if type_txt == "agreement":
            return "Agreement"
        return type_txt

    overall_df["name"] = overall_df["name"].apply(name_mapping)
    overall_df["dimension"] = overall_df["dimension"].apply(name_mapping)
    overall_df["type"] = overall_df["type"].apply(type_mapping)

    iou_df = overall_df[overall_df["type"] == "IoU"]
    diversity_df = overall_df[overall_df["type"] != "IoU"]

    barplot(iou_df, base_dir, prefix="IoU")
    barplot(diversity_df, base_dir, prefix="Diversity")


def combined_details(base_dir, iou_details_df, agreement_details_df,
                     correlation_details_df):
    iou_df = iou_details_df[iou_details_df["class"] != "overall"].copy(
        deep=True)
    agreement_df = agreement_details_df[
        agreement_details_df["class"] != "overall"].copy(deep=True)
    correlation_df = correlation_details_df[
        correlation_details_df["class"] != "overall"].copy(deep=True)

    iou_df["type"] = "iou"
    agreement_df["type"] = "agreement"
    correlation_df["type"] = "correlation"

    iou_df = iou_df[["name", "score", "type", "dimension"]]
    agreement_df = agreement_df[["name", "score", "type", "dimension"]]
    correlation_df = correlation_df[["name", "score", "type", "dimension"]]

    combined_df = pd.concat([iou_df, agreement_df,
                             correlation_df]).copy(deep=True)

    def name_mapping(name):
        return name.replace("_", " ").title()

    def type_mapping(type_txt):
        if type_txt == "iou":
            return "IoU"
        if type_txt == "correlation":
            return "Correlation"
        if type_txt == "agreement":
            return "Agreement"
        return type_txt

    combined_df["name"] = combined_df["name"].apply(name_mapping)
    combined_df["dimension"] = combined_df["dimension"].apply(name_mapping)
    combined_df["type"] = combined_df["type"].apply(type_mapping)

    iou_df = combined_df[combined_df["type"] == "IoU"]
    diversity_df = combined_df[combined_df["type"] != "IoU"]

    boxplot(iou_df, base_dir, prefix="IoU")
    boxplot(diversity_df, base_dir, prefix="Diversity")


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

    overall_scores(ebm_dir, iou_scores_df, agreement_scores_df,
                   correlation_scores_df)
    overall_details(ebm_dir, iou_details_df, agreement_details_df,
                    correlation_details_df)
    combined_details(ebm_dir, iou_details_df, agreement_details_df,
                     correlation_details_df)
