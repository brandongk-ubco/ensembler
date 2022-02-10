import torch
import numpy as np
import os


def RoundUp(x, mul):
    return ((x + mul - 1) & (-mul))


def weighted_loss(y_hat, y, weights, loss_function):
    loss = 0
    assert len(weights) == y_hat.shape[1]
    for i, w in enumerate(weights):
        i_y = y[:, i, :, :]
        i_y_hat = y_hat[:, i, :, :]
        loss += loss_function(i_y_hat.clone(), i_y.clone()).mean() * w
    loss = loss.mean()
    return loss


def crop_image_only_outside(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    img = img.max(axis=2)
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return row_start, row_end, col_start, col_end


def classwise(y_hat, y, metric):
    assert type(y_hat) == type(y)
    if torch.is_tensor(y_hat):
        return classwise_torch(y_hat, y, metric)
    return classwise_numpy(np.asarray(y_hat), np.asarray(y), metric)


def classwise_numpy(y_hat, y, metric):

    assert y_hat.shape == y.shape

    results = np.empty(y_hat.shape[2], dtype=y_hat.dtype)

    for i in range(y_hat.shape[2]):
        y_hat_class = y_hat[:, :, i]
        y_class = y[:, :, i]
        assert y_hat_class.shape == y_class.shape
        results[i] = metric(y_hat_class.flatten(), y_class.flatten())

    return results


def classwise_torch(y_hat, y, metric):

    assert y_hat.shape == y.shape

    results = torch.empty(y_hat.shape[1],
                          dtype=y_hat.dtype,
                          device=y_hat.device)

    for i in torch.tensor(range(y_hat.shape[1]),
                          dtype=torch.long,
                          device=y_hat.device):
        y_hat_class = y_hat.index_select(1, i)
        y_class = y.index_select(1, i)
        results[i] = metric(y_hat_class, y_class)

    return results


def extract_explanation(ebm, out_dir):

    rows = []
    for i, name in enumerate(ebm.data()["names"]):
        rows.append({
            "name": name,
            "score": ebm.data()["scores"][i],
            "dimension": "importance",
            "upper_bounds": None,
            "lower_bounds": None
        })

    for index, feature_name in enumerate(ebm.feature_names):

        feature_type = ebm.feature_types[index]

        plotly_fig = ebm.visualize(index)
        outfile = os.path.join(out_dir, f"{feature_name}.png")
        plotly_fig.write_image(outfile)

        data = ebm.data(index)
        if feature_type == "categorical":
            for i, name in enumerate(data["names"]):
                rows.append({
                    "name": name,
                    "score": data["scores"][i],
                    "dimension": feature_name,
                    "upper_bounds": data["upper_bounds"][i],
                    "lower_bounds": data["lower_bounds"][i]
                })
        elif feature_type == "continuous":
            names = data["names"]
            for i in range(len(names) - 1):
                rows.append({
                    "name": f"{names[i]} - {names[i + 1]}",
                    "score": data["scores"][i],
                    "dimension": feature_name,
                    "upper_bounds": data["upper_bounds"][i],
                    "lower_bounds": data["lower_bounds"][i]
                })
        elif feature_type == "interaction":
            # TODO: Explain here
            pass
        else:
            raise ValueError(f"Unknonwn feature tyupe {feature_type}")

    return rows
