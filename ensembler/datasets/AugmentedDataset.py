import numpy as np
import torch
import random
from ensembler.utils import crop_image_only_outside
from math import ceil
from functools import lru_cache


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


class RepeatedDatasetAugmenter(AugmentedDataset):
    def __init__(self,
                 dataset,
                 patch_transform,
                 augments=None,
                 preprocessing_transform=None,
                 shuffle=False,
                 min_train_samples=200,
                 **kwargs):
        super().__init__(dataset, preprocessing_transform, patch_transform,
                         augments)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle
        self.augments = augments
        self.repeats = 1
        if num_elements < min_train_samples:
            self.repeats = ceil(min_train_samples / num_elements)

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):

        repeat_idx = idx % len(self.dataset)

        if repeat_idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        image, mask = super().__getitem__(self.data_map[idx])

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

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


class RepeatedBatchDatasetAugmenter(AugmentedDataset):
    def __init__(self,
                 dataset,
                 patch_transform,
                 augments=None,
                 preprocessing_transform=None,
                 shuffle=False,
                 min_train_samples=200,
                 **kwargs):
        super().__init__(dataset, preprocessing_transform, patch_transform,
                         augments)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle
        self.augments = augments
        self.repeats = 1
        if num_elements < min_train_samples:
            self.repeats = ceil(min_train_samples / num_elements)

    def __len__(self):
        return len(self.dataset) * self.repeats * 4

    @lru_cache(maxsize=10)
    def get_dataset_img(self, dataset_idx):
        return super().__getitem__(dataset_idx)

    def __getitem__(self, idx):

        repeat_idx = idx % (4 * len(self.dataset))

        if repeat_idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        img_idx = repeat_idx // 4

        image, mask = self.get_dataset_img(self.data_map[img_idx])

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        flip_idx = repeat_idx % 4

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if flip_idx == 1:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])

        if flip_idx == 2:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])

        if flip_idx == 3:
            image = torch.flip(image, [1, 2])
            mask = torch.flip(mask, [1, 2])

        return image, mask


class BatchDatasetAugmenter(AugmentedDataset):
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

    @lru_cache(maxsize=10)
    def get_dataset_img(self, dataset_idx):
        return super().__getitem__(dataset_idx)

    def __len__(self):
        return len(self.dataset) * 4

    def __getitem__(self, idx):

        if idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        img_idx = idx // 4
        flip_idx = idx % 4

        image, mask = self.get_dataset_img(self.data_map[img_idx])

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if flip_idx == 1:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])

        if flip_idx == 2:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])

        if flip_idx == 3:
            image = torch.flip(image, [1, 2])
            mask = torch.flip(mask, [1, 2])

        return image, mask
