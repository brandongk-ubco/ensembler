import matplotlib
import numpy as np
import os
from matplotlib import pyplot as plt
from p_tqdm import p_umap as mapper
import glob

matplotlib.use('Agg')

outdir = "/mnt/d/work/repos/ensembler/lightning_logs/version_2/predictions/"


def visualize_prediction(src):
    prediction = np.load(src)
    predicted_mask = prediction["prediction"].transpose(1, 2, 0)
    image = prediction["image"].transpose(1, 2, 0)
    mask = prediction["mask"].transpose(1, 2, 0)

    filename = os.path.basename(src)
    name, ext = os.path.splitext(filename)

    intensity = 255 // mask.shape[2]
    mask_img = np.argmax(mask, axis=2) * intensity

    predicted_mask_img = np.argmax(predicted_mask, axis=2) * intensity

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    if len(image.shape) == 2:
        ax1.imshow(image, cmap="gray")
    else:
        ax1.imshow(image, cmap="gray")

    ax2.imshow(mask_img, cmap="gray", vmin=0, vmax=255)
    ax3.imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.savefig(os.path.join(outdir, "{}.png".format(name)))
    plt.close()


if __name__ == '__main__':

    predictions = glob.glob(os.path.join(outdir, "*.npz"))
    mapper(visualize_prediction, predictions)
