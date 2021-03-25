import pandas as pd


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
