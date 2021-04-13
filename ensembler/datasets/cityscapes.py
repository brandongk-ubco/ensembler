from .AugmentedDataset import AugmentedDataset
import torch
import numpy as np
import torchvision

num_classes = 20

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
loss_weights = [1.] * num_classes
loss_weights[0] = 0.

cityscapes_folder = "/mnt/d/work/datasets/cityscapes"


class CityscapesAugmentedDataset(AugmentedDataset):

    classes = [
        "background", "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle"
    ]

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)

        image = np.array(image)
        mask = np.array(mask)

        label_mask = np.zeros(
            (len(self.classes), mask.shape[0], mask.shape[1]),
            dtype=image.dtype)

        for k, v in mapping_20.items():
            label_mask[v, mask == k] = 1

        image = image.astype("float32").swapaxes(0, 2).swapaxes(1, 2) / 255.

        label_mask = torch.Tensor(label_mask)
        image = torch.Tensor(image)

        return image, label_mask


def get_dataloaders(augmentations):
    train_transform, val_transform, test_transform = augmentations
    train_data = torchvision.datasets.Cityscapes(cityscapes_folder,
                                                 split='train',
                                                 mode='fine',
                                                 target_type='semantic')

    val_data = torchvision.datasets.Cityscapes(cityscapes_folder,
                                               split='val',
                                               mode='fine',
                                               target_type='semantic')

    test_data = torchvision.datasets.Cityscapes(cityscapes_folder,
                                                split='test',
                                                mode='fine',
                                                target_type='semantic')

    train_data = CityscapesAugmentedDataset(train_data, train_transform)
    val_data = CityscapesAugmentedDataset(val_data, val_transform)
    test_data = CityscapesAugmentedDataset(test_data, test_transform)

    return train_data, val_data, test_data
