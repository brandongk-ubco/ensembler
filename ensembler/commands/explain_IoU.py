import os
import pandas as pd
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.perf import RegressionPerf


def explain(df, base_dir, random_state=42, test_size=0.50):
    train_cols = [
        'depth', 'residual_units', 'width', 'width_ratio', 'activation', 'class'
    ]
    label = "value"
    if len(df["class"].unique()) > 1:
        train_cols.append("class")
    label = "agreement"

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
    ebm_global = ebm.explain_global(name="IoU Predictor")

    ebm_perf = RegressionPerf(ebm.predict).explain_perf(X_test,
                                                        y_test,
                                                        name='IoU Prediction')

    ebm_dir = os.path.join(base_dir, "mIoU")
    os.makedirs(ebm_dir, exist_ok=True)

    plotly_fig = ebm_global.visualize()
    outfile = os.path.join(ebm_dir, "importance.png")
    plotly_fig.write_image(outfile)

    rows = []
    for i, name in enumerate(ebm_global.data()["names"]):
        rows.append({
            "name": name,
            "score": ebm_global.data()["scores"][i],
            "type": "IoU",
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
                "type": "IoU",
                "dimension": feature_name,
                "upper_bounds": data["upper_bounds"][i],
                "lower_bounds": data["lower_bounds"][i]
            })

    plotly_fig = ebm_perf.visualize()
    outfile = os.path.join(ebm_dir, "agreement.png")
    plotly_fig.write_image(outfile)

    return score, rows


def explain_IoU(base_dir: str):

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

    for split_type in iou_df["type"].unique():
        split_df = iou_df[iou_df["type"] == split_type].copy(deep=True)
        split_dir = os.path.join(base_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        explain_performance(split_df, split_dir)
