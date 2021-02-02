import pytorch_lightning as pl
import matplotlib
from data import get_dataloaders
import numpy as np
import os
from matplotlib import pyplot as plt
from p_tqdm import p_umap as mapper
pl.seed_everything(42)
matplotlib.use('Agg')

datapath = "/mnt/d/work/datasets/wrinkler/"
outdir = "/mnt/d/work/repos/wrinkler/lightning_logs/version_7/predictions/"


def visualize_prediction(src):
    i, x, y = src
    name = os.path.splitext(
        (os.path.basename(test_loader.dataset.dataset.images[i])))[0]
    y_hat = np.load(os.path.join(outdir, "{}.npz".format(name)))["prediction"]
    predicted_mask = np.round(y_hat, 0)

    mask = y.squeeze(0).numpy()
    img = x.squeeze(0).numpy()
    mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
    mask_img[mask[1, :, :] == 1] = 50
    mask_img[mask[2, :, :] == 1] = 100
    mask_img[mask[3, :, :] == 1] = 200

    predicted_mask_img = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
    predicted_mask_img[predicted_mask[1, :, :] > 0.5] = 50
    predicted_mask_img[predicted_mask[2, :, :] > 0.5] = 100
    predicted_mask_img[predicted_mask[3, :, :] > 0.5] = 200

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
    ax2.imshow(mask_img)
    ax3.imshow(predicted_mask_img)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.savefig(os.path.join(outdir, "{}.png".format(name)))
    plt.close()


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_dataloaders(datapath,
                                                            use_cache=False)

    mapper(visualize_prediction,
           [(i, x, y) for i, (x, y) in enumerate(test_loader)])
