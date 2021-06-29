import torch


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


def classwise(y_hat, y, metric, dim=1):

    results = torch.empty(y_hat.shape[dim],
                          dtype=y_hat.dtype,
                          device=y_hat.device)

    for i in torch.tensor(range(y_hat.shape[dim]),
                          dtype=torch.long,
                          device=y_hat.device):
        y_hat_class = y_hat.index_select(dim, i)
        y_class = y.index_select(dim, i)
        results[i] = metric(y_hat_class, y_class)

    return results
