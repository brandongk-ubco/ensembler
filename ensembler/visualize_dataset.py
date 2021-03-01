from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from train import dataset, get_augments, model

matplotlib.use('Agg')

if __name__ == '__main__':

    m = model(dataset, get_augments, patience=10)

    intensity = 255 // dataset.num_classes

    for j, (imgs, masks) in enumerate(m.train_dataloader()):
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()
            mask = masks[i, :, :, :].numpy()

            mask_img = np.argmax(mask, axis=0) * intensity

            fig, (ax1, ax2) = plt.subplots(2, 1)

            if img.shape[0] == 1:
                ax1.imshow(img.squeeze(), cmap="gray")
            else:
                ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1))
            ax2.imshow(mask_img, cmap="gray", vmin=0, vmax=255)

            ax1.axis('off')
            ax2.axis('off')

            plt.savefig("{}-{}.png".format(j, i))
            plt.close()
            input("Press ENTER to continue")
