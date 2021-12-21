from ensembler.methods import EnsembleMethods
from argh import arg
import pandas as pd
import os


@arg('method_name', choices=EnsembleMethods.choices())
def ensemble(in_dir: str, method_name: EnsembleMethods):
    assert method_name is not None
    method = EnsembleMethods.get(method_name)

    metrics_file = os.path.join(in_dir, "metrics.csv")
    diversity_file = os.path.join(in_dir, "diversity.csv")
    tukey_file = os.path.join(in_dir, "tukey.csv")
    assert os.path.exists(metrics_file)
    assert os.path.exists(diversity_file)
    assert os.path.exists(tukey_file)

    metrics = pd.read_csv(metrics_file)
    diversity = pd.read_csv(diversity_file)
    tukey = pd.read_csv(tukey_file)

    mIoU = metrics[metrics["metric"] == "iou"][[
        "job_hash", "value"
    ]].groupby(by="job_hash").mean().rename(columns={
        "value": "mIoU"
    }).sort_values(by="mIoU", ascending=False)

    import pdb
    pdb.set_trace()
