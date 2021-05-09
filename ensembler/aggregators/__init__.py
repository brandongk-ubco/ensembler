import torch


def harmonize_batch(y_hat,
                    y,
                    reduction=lambda y_hat: 1 - torch.prod(1 - y_hat, dim=0),
                    activation=torch.nn.Softmax2d()):

    assert torch.max(y_hat) >= 0
    assert torch.max(y_hat) <= 1
    assert torch.max(y) >= 0
    assert torch.max(y) <= 1

    y_hat = y_hat.clone()
    y = y.clone()

    y_hat[1, :, :, :] = torch.flip(y_hat[1, :, :, :], [1])
    y_hat[2, :, :, :] = torch.flip(y_hat[2, :, :, :], [2])
    y_hat[3, :, :, :] = torch.flip(y_hat[3, :, :, :], [1, 2])
    y[1, :, :, :] = torch.flip(y[1, :, :, :], [1])
    y[2, :, :, :] = torch.flip(y[2, :, :, :], [2])
    y[3, :, :, :] = torch.flip(y[3, :, :, :], [1, 2])

    assert torch.allclose(y[1, :, :, :], y[0, :, :, :])
    assert torch.allclose(y[2, :, :, :], y[0, :, :, :])
    assert torch.allclose(y[3, :, :, :], y[0, :, :, :])

    y_hat = reduction(y_hat)
    y = y[0, :, :, :]

    assert torch.max(y_hat) >= 0
    assert torch.max(y_hat) <= 1
    assert torch.max(y) >= 0
    assert torch.max(y) <= 1

    return y_hat, y
