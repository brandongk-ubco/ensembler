import torch


def harmonize_batch(y_hat,
                    y,
                    reduction=lambda y_hat: 1 - torch.prod(1 - y_hat, dim=0),
                    activation=torch.nn.Softmax2d()):

    y_hat = y_hat.clone()
    y = y.clone()

    y_hat[1, :, :, :] = torch.flip(y_hat[1, :, :, :], [1])
    y_hat[2, :, :, :] = torch.flip(y_hat[2, :, :, :], [2])
    y_hat[3, :, :, :] = torch.flip(y_hat[3, :, :, :], [1, 2])
    y[1, :, :, :] = torch.flip(y[1, :, :, :], [1])
    y[2, :, :, :] = torch.flip(y[2, :, :, :], [2])
    y[3, :, :, :] = torch.flip(y[3, :, :, :], [1, 2])

    eps = torch.finfo(y.dtype).eps
    assert torch.all(y[0, :, :, :] - y[1, :, :, :] <= eps)
    assert torch.all(y[0, :, :, :] - y[2, :, :, :] <= eps)
    assert torch.all(y[0, :, :, :] - y[3, :, :, :] <= eps)
    assert torch.all(y[1, :, :, :] - y[0, :, :, :] <= eps)
    assert torch.all(y[2, :, :, :] - y[0, :, :, :] <= eps)
    assert torch.all(y[3, :, :, :] - y[0, :, :, :] <= eps)

    y_hat = reduction(y_hat)
    # y_hat = torch.unsqueeze(y_hat, 0)
    # y_hat = activation(y_hat)
    # y_hat = y_hat[0, :, :, :]
    y = y[0, :, :, :]

    return y_hat, y
