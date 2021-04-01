import segmentation_models_pytorch as smp
import torch


def batch_loss(y_hat,
               y,
               reduction=lambda y_hat: 1 - torch.prod(1 - y_hat, dim=0),
               loss=smp.losses.FocalLoss("multilabel")):

    batches = y_hat.shape[0]
    loss_val = 0
    for batch in range(0, batches, 4): #Should be batch size
        y_hat_batch = y_hat[batch:batch + 4, :, :, :].clone()
        y_hat_batch[1, :, :, :] = torch.flip(y_hat_batch[1, :, :, :].clone(),
                                             [1])
        y_hat_batch[2, :, :, :] = torch.flip(y_hat_batch[2, :, :, :].clone(),
                                             [2])
        y_hat_batch[3, :, :, :] = torch.flip(y_hat_batch[3, :, :, :].clone(),
                                             [1, 2])
        y_batch = y[batch:batch + 4, :, :, :].clone()
        y_batch[1, :, :, :] = torch.flip(y_batch[1, :, :, :].clone(), [1])
        y_batch[2, :, :, :] = torch.flip(y_batch[2, :, :, :].clone(), [2])
        y_batch[3, :, :, :] = torch.flip(y_batch[3, :, :, :].clone(), [1, 2])

        try:
            eps = torch.finfo(y.dtype).eps
            assert torch.all(y_batch[0, :, :, :] - y_batch[1, :, :, :] <= eps)
            assert torch.all(y_batch[0, :, :, :] - y_batch[2, :, :, :] <= eps)
            assert torch.all(y_batch[0, :, :, :] - y_batch[3, :, :, :] <= eps)
            assert torch.all(y_batch[1, :, :, :] - y_batch[0, :, :, :] <= eps)
            assert torch.all(y_batch[2, :, :, :] - y_batch[0, :, :, :] <= eps)
            assert torch.all(y_batch[3, :, :, :] - y_batch[0, :, :, :] <= eps)
        except AssertionError:
            print("Warning: Batch masks are not equal!")
            print(torch.max(y_batch[0, :, :, :] - y_batch[1, :, :, :]))
            print(torch.max(y_batch[0, :, :, :] - y_batch[2, :, :, :]))
            print(torch.max(y_batch[0, :, :, :] - y_batch[3, :, :, :]))

        y_hat_batch = reduction(y_hat_batch)
        y_batch = y_batch[0, :, :, :]

        loss_val += loss(y_hat_batch, y_batch)

    return loss_val
