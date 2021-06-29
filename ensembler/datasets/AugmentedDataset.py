import numpy as np
import torch
import random
from ensembler.utils import crop_image_only_outside


class AugmentedDataset:
    def __init__(self, dataset, preprocessing_transform, patch_transform,
                 augment_transform):
        self.dataset = dataset
        self.patch_transform = patch_transform
        self.augment_transform = augment_transform
        self.preprocessing_transform = preprocessing_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)
        image = np.array(image)
        mask = np.array(mask)

        row_start, row_end, col_start, col_end = crop_image_only_outside(
            image, tol=0.2)
        image = image[row_start:row_end, col_start:col_end, :]
        mask = mask[row_start:row_end, col_start:col_end, :]

        if self.preprocessing_transform is not None:
            transformed = self.preprocessing_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.patch_transform is not None:
            transformed = self.patch_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        expected_shape = image.shape

        if self.augment_transform is not None:
            transformed = self.augment_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.patch_transform is not None and expected_shape != image.shape:
            transformed = self.patch_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.clip(image, 0., 1.)
        mask = mask.astype(image.dtype)

        return image, mask


class DatasetAugmenter(AugmentedDataset):
    def __init__(self,
                 dataset,
                 patch_transform,
                 augments=None,
                 preprocessing_transform=None,
                 shuffle=False,
                 **kwargs):
        super().__init__(dataset, preprocessing_transform, patch_transform,
                         augments)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle
        self.augments = augments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        image, mask = super().__getitem__(self.data_map[idx])

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask
