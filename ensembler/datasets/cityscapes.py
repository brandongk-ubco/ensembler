import os
from .AugmentedDataset import AugmentedDataset
import torch
import numpy as np
import torchvision

mapping_20 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0
}

cityscapes_folder = os.environ["CITYSCAPES_DATASET"]

train_data = torchvision.datasets.Cityscapes(cityscapes_folder,
                                             split='train',
                                             mode='fine',
                                             target_type='semantic')

val_data = torchvision.datasets.Cityscapes(cityscapes_folder,
                                           split='val',
                                           mode='fine',
                                           target_type='semantic')


class CityscapesAugmentedDataset(AugmentedDataset):

    classes = [
        "background", "road", "sidewalk", "building", "wall", "fence",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle", "license plate"
    ]

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)

        label_mask = np.zeros((20, mask.shape[0], mask.shape[1]),
                              dtype=image.dtype)

        for k, v in mapping_20.items():
            label_mask[v, mask == k] = 1

        label_mask = torch.Tensor(label_mask)

        return image, label_mask
