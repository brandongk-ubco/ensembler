import torch
import numpy as np
import torchvision
from PIL import Image
import os

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


class CityscapesDatasetInitializer:
    def __init__(self, cityscapes_folder):

        self.cityscapes_folder = cityscapes_folder

        self.targets_dir = os.path.join(cityscapes_folder, "gtFine")

        train_data = torchvision.datasets.Cityscapes(self.cityscapes_folder,
                                                     split='train',
                                                     mode='fine',
                                                     target_type='semantic')

        val_data = torchvision.datasets.Cityscapes(self.cityscapes_folder,
                                                   split='val',
                                                   mode='fine',
                                                   target_type='semantic')

        test_data = torchvision.datasets.Cityscapes(self.cityscapes_folder,
                                                    split='test',
                                                    mode='fine',
                                                    target_type='semantic')

        sep = os.path.sep
        all_images = train_data.images + val_data.images + test_data.images
        all_images = [sep.join(i.split(sep)[-3:]) for i in all_images]

        all_masks = [
            os.path.join(
                "gtFine", '{}_gtFine_labelIds.png'.format(
                    file_name.split('_leftImg8bit')[0]))
            for file_name in all_images
        ]

        all_images = [os.path.join("leftImg8bit", i) for i in all_images]

        self.samples = [i for i in zip(all_images, all_masks)]

    def get_image_names(self):
        return [
            os.path.splitext(os.path.basename(i))[0].replace(
                "_leftImg8bit", "") for i, m in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]

        image = Image.open(os.path.join(self.cityscapes_folder,
                                        image_name)).convert('RGB')
        mask = Image.open(os.path.join(self.cityscapes_folder, mask_name))

        image = np.array(image)
        mask = np.array(mask)

        label_mask = np.zeros((20, mask.shape[0], mask.shape[1]),
                              dtype=image.dtype)

        for k, v in mapping_20.items():
            label_mask[v, mask == k] = 1

        image = image.astype("float32") / 255.
        mask = mask.transpose(1, 2, 0)

        label_mask = torch.Tensor(label_mask)
        image = torch.Tensor(image)

        return image, label_mask


def get_all_dataloader(directory):
    return CityscapesDatasetInitializer(directory)
