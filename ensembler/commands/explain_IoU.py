import os
import pandas as pd
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.perf import RegressionPerf
from ensembler.utils import extract_explanation


def explain(df, ebm_dir, random_state=42, test_size=0.50):
    train_cols = [
        'depth', 'residual_units', 'width', 'width_ratio', 'activation'
    ]
    label = "value"
    if len(df["class"].unique()) > 1:
        train_cols.append("class")

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

    os.makedirs(ebm_dir, exist_ok=True)

    plotly_fig = ebm_global.visualize()
    outfile = os.path.join(ebm_dir, "importance.png")
    plotly_fig.write_image(outfile)

    rows = extract_explanation(ebm_global, ebm_dir)

    plotly_fig = ebm_perf.visualize()
    outfile = os.path.join(ebm_dir, "performance.png")
    plotly_fig.write_image(outfile)

    return score, rows


def build_df(in_dir: str):

    metrics_file = os.path.join(in_dir, "metrics.csv")

    assert os.path.exists(metrics_file)

    df = pd.read_csv(metrics_file)
    df = df[df["type"] == "test"]

    grouped = df.groupby(by=[
        "job_hash", "class", "metric", "model_activation", "model_width",
        "model_width_ratio", "type"
    ]).mean().reset_index()

    iou_df = grouped[grouped.metric == "iou"].sort_values("value")

    def renaming_fun(x):
        return x.replace("model_", "")

    iou_df = iou_df.rename(columns=renaming_fun)

    return iou_df


def explain_IoU(in_dir: str):

    df = build_df(in_dir)

    scores = []
    details = []

    print("Overall")

    score, rows = explain(df, os.path.join(in_dir, "ebms", "IoU", "overall"))

    rows = [dict(**a, **{"class": "overall"}) for a in rows]

    details += rows

    scores.append({"class": "overall", "score": score})

    classes = df["class"].unique()
    for clazz in classes:
        print(clazz)
        class_df = df[df["class"] == clazz]
        score, rows = explain(class_df,
                              os.path.join(in_dir, "ebms", "IoU", clazz))

        rows = [dict(**a, **{"class": clazz}) for a in rows]

        details += rows
        scores.append({
            "class": clazz,
            "score": score,
        })

    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(in_dir, "ebms", "IoU", "scores.csv"), index=False)

    df = pd.DataFrame(details)
    df.to_csv(os.path.join(in_dir, "ebms", "IoU", "details.csv"), index=False)
