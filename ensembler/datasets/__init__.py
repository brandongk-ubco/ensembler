import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm


def visualize_dataset(dataset, classes, color_map=None, n=None):

    if color_map is None:
        color_map = cm.get_cmap("tab20").colors

    for idx, (image, mask) in enumerate(dataset):
        print(idx)

        image = image.numpy()
        image -= image.min()
        image *= 255 / image.max()
        image = image.astype("uint8")
        image = np.moveaxis(image, 0, -1)
        mask = mask.numpy()
        mask = np.moveaxis(mask, 0, -1)

        highlighted_image = image.copy()

        for i in range(len(classes)):
            print(i)
            mask_i = mask[:, :, i].copy().astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_i, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            color = [int(x * 255) for x in color_map[i][:3]]
            cv2.drawContours(highlighted_image, contours, -1, color, 3)

        # legend = [
        #     Line2D(
        #         [0],
        #         [0],
        #         color=color,
        #         lw=4,
        #     ) for color in color_map[0:len(classes)]
        # ]

        fig, ax = plt.subplots()
        # ax.imshow(np.moveaxis(image, 0, -1), interpolation='none')
        ax.imshow(highlighted_image, alpha=1, interpolation='none')
        ax.axis('off')

        # plt.legend(legend,
        #            classes,
        #            bbox_to_anchor=(0, 1, 1, 0.2),
        #            loc="lower left",
        #            frameon=False,
        #            ncol=len(classes))

        plt.savefig("%s.png" % idx, dpi=150, bbox_inches='tight')

        plt.close()
