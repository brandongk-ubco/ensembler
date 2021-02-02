import pytorch_lightning as pl
import matplotlib
from data import get_dataloaders
from Model import Segmenter
import numpy as np
import torch
import os
from p_tqdm import t_map as mapper

pl.seed_everything(42)
matplotlib.use('Agg')

datapath = "/mnt/d/work/datasets/wrinkler/"
model = "/mnt/d/work/repos/wrinkler/lightning_logs/version_7/checkpoints/epoch=10-step=912.ckpt"
outdir = "/mnt/d/work/repos/wrinkler/lightning_logs/version_7/predictions/"

os.makedirs(outdir, exist_ok=True)


def getNextMultiple(x, multiple):
    return ((x + multiple - 1) & (-multiple))


def padImageToMultiple(img, multiple=256):
    img = img.numpy().squeeze(0)
    ww = getNextMultiple(img.shape[2], multiple)
    hh = getNextMultiple(img.shape[1], multiple)
    xx = (ww - img.shape[2]) // 2
    yy = (hh - img.shape[1]) // 2

    img = np.pad(img, ((0, 0), (yy, yy), (xx, xx)), 'constant')
    return torch.from_numpy(np.expand_dims(img, 0))


def unpadImage(img, padded):
    yy = (padded.shape[2] - img.shape[2]) // 2
    xx = (padded.shape[3] - img.shape[3]) // 2
    return padded[:, :, yy:padded.shape[2] - yy, xx:padded.shape[3] - xx]


def predict(src):
    i, x, y = src
    name = os.path.splitext(
        (os.path.basename(test_loader.dataset.dataset.images[i])))[0]
    x_pad = padImageToMultiple(x)
    y_hat = wrinkler(x_pad).numpy()
    y_hat = unpadImage(y, y_hat)
    assert y_hat.shape == y.numpy().shape
    y_hat = y_hat.squeeze(0)
    np.savez_compressed(os.path.join(outdir, name), prediction=y_hat)


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(datapath,
                                                            use_cache=False)

    wrinkler = Segmenter.load_from_checkpoint(model)
    wrinkler.freeze()
    mapper(predict, [(i, x, y) for i, (x, y) in enumerate(test_loader)])
