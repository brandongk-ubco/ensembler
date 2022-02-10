import pandas as pd
import os
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.perf import RegressionPerf


def explain(df, base_dir, random_state=42, test_size=0.50):
    train_cols = [
        'depth', 'residual_units', 'width', 'width_ratio', 'activation',
        "performance_diff"
    ]
    if len(df["class"].unique()) > 1:
        train_cols.append("class")
    label = "disagreement_correlation"

    X = df[train_cols]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    feature_types = ['categorical'] * len(train_cols)

    ebm = ExplainableBoostingRegressor(
        feature_names=train_cols,
        feature_types=feature_types,
        # Overall
        n_jobs=os.cpu_count(),
        random_state=random_state,
    )

    print("Fitting EBM")
    ebm.fit(X_train, y_train)
    score = ebm.score(X_test, y_test)
    print("Score: {}".format(score))
    ebm_global = ebm.explain_global(name="Correlation Predictor")

    ebm_perf = RegressionPerf(ebm.predict).explain_perf(
        X_test, y_test, name='Correlation Prediction')

    ebm_dir = os.path.join(base_dir, "correlation")
    os.makedirs(ebm_dir, exist_ok=True)

    plotly_fig = ebm_global.visualize()
    outfile = os.path.join(ebm_dir, "importance.png")
    plotly_fig.write_image(outfile)

    rows = []
    for i, name in enumerate(ebm_global.data()["names"]):
        rows.append({
            "name": name,
            "score": ebm_global.data()["scores"][i],
            "type": "correlation",
            "dimension": "importance",
            "upper_bounds": None,
            "lower_bounds": None
        })

    for index, feature_name in enumerate(ebm_global.feature_names):
        plotly_fig = ebm_global.visualize(index)
        outfile = os.path.join(ebm_dir, f"{feature_name}.png")
        plotly_fig.write_image(outfile)

        data = ebm_global.data(index)
        if "names" not in data:
            continue
        for i, name in enumerate(data["names"]):
            if data["type"] != "univariate":
                continue
            rows.append({
                "name": name,
                "score": data["scores"][i],
                "type": "correlation",
                "dimension": feature_name,
                "upper_bounds": data["upper_bounds"][i],
                "lower_bounds": data["lower_bounds"][i]
            })

    plotly_fig = ebm_perf.visualize()
    outfile = os.path.join(ebm_dir, "correlation.png")
    plotly_fig.write_image(outfile)

    return score, rows


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

    return diversity


def explain_correlation(in_dir: str):

    diversity = build_df(in_dir)

    scores = []
    details = []

    print("Overall")

    correlation_score, correlation_rows = explain(
        diversity, os.path.join(in_dir, "ebms", "correlation", "overall"))

    correlation_rows = [
        dict(**a, **{"class": "overall"}) for a in correlation_rows
    ]

    details += correlation_rows

    scores.append({"class": "overall", "correlation_score": correlation_score})

    classes = diversity["class"].unique()
    for clazz in classes:
        print(clazz)
        class_df = diversity[diversity["class"] == clazz]
        correlation_score, correlation_rows = explain(
            class_df, os.path.join(in_dir, "ebms", "correlation", clazz))

        correlation_rows = [
            dict(**a, **{"class": clazz}) for a in correlation_rows
        ]

        details += correlation_rows
        scores.append({
            "class": clazz,
            "correlation_score": correlation_score,
        })

    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(in_dir, "ebms", "correlation_scores.csv"),
              index=False)

    df = pd.DataFrame(details)
    df.to_csv(os.path.join(in_dir, "ebms", "correlation_details.csv"),
              index=False)
