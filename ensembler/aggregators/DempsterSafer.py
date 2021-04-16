import torch
from typing import List


def ds_combination(sensor_one, sensor_two):
    assert sensor_one.shape == sensor_two.shape
    assert len(sensor_one.shape) == 1
    num_elems = sensor_one.shape[0]
    tiled_sensor_one = sensor_one.repeat(num_elems, 1)
    tiled_sensor_two = sensor_two.repeat(num_elems, 1).transpose(0, 1)
    multiplied = tiled_sensor_one * tiled_sensor_two
    belief = torch.diagonal(multiplied)
    eps = 1e-7

    if torch.all(belief < eps):
        return belief

    disbelief = 1 - (torch.sum(multiplied) - torch.sum(belief))
    return belief / disbelief


def ds_reduce_column(x: int, y: int, y_hat, reduced) -> torch.Tensor:
    for k in range(1, y_hat.shape[0]):
        reduced[:, x, y] = ds_combination(reduced[:, x, y], y_hat[k, :, x, y])
    return torch.empty(0)


@torch.jit.script
def ds_reduce(y_hat):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    reduced = y_hat[0, :, :, :].clone()
    expected = y_hat.shape[2] * y_hat.shape[3]

    i = 0

    for x in range(y_hat.shape[2]):
        for y in range(y_hat.shape[3]):
            if len(futures) > 100:
                for x, future in enumerate(futures):
                    if i % 10000 == 0:
                        print(i / expected)
                    i += 1
                    torch.jit.wait(future)
                futures.clear()
            futures.append(
                torch.jit.fork(ds_reduce_column, x, y, y_hat, reduced))

    return reduced
