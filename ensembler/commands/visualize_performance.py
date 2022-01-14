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


def visualize_performance(base_dir: str):

    metrics_file = os.path.join(base_dir, "metrics.csv")

    assert os.path.exists(metrics_file)

    random_state = 42
    test_size = 0.50

    df = pd.read_csv(metrics_file)
    grouped = df.groupby(by=["job_hash", "class", "metric", "model_activation"
                            ]).mean().reset_index()

    iou_df = grouped[grouped.metric == "iou"].sort_values("value")

    def renaming_fun(x):
        return x.replace("model_", "")

    iou_df = iou_df.rename(columns=renaming_fun)

    iou_df["width_ratio"] = (iou_df["width_ratio"] * 10).astype(int)

    plot = sns.boxplot(x='class', y='value', data=iou_df, color="skyblue")

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

    train_cols = [
        'depth', 'residual_units', 'width', 'width_ratio', 'activation', 'class'
    ]
    label = "value"

    X = iou_df[train_cols]
    y = iou_df[label]

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

    # ebm_local = ebm.explain_local(X_test, y_test, name="Local IoU Predictor")
    ebm_perf = RegressionPerf(ebm.predict).explain_perf(X_test,
                                                        y_test,
                                                        name='IoU Prediction')

    ebm_dir = os.path.join(base_dir, "ebm")
    os.makedirs(ebm_dir, exist_ok=True)

    plotly_fig = ebm_global.visualize()
    outfile = os.path.join(ebm_dir, "importance.png")
    plotly_fig.write_image(outfile)

    for index, feature_name in enumerate(ebm_global.feature_names):
        plotly_fig = ebm_global.visualize(index)
        outfile = os.path.join(ebm_dir, f"{feature_name}.png")
        plotly_fig.write_image(outfile)

    plotly_fig = ebm_perf.visualize()
    outfile = os.path.join(ebm_dir, "performance.png")
    plotly_fig.write_image(outfile)
