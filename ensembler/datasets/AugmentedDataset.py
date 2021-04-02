import numpy as np
import torch
from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure
from skimage.util import dtype_limits
import random


def contrast_stretch(image, min_percentile=2, max_percentile=98):
    p2, p98 = np.percentile(image, (min_percentile, max_percentile))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    min_val, max_val = dtype_limits(image, clip_negative=True)
    image -= np.min(image)
    image /= np.max(image)
    image = np.clip(image, 0., 1.)
    return image


class AugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)
        image = np.array(image)
        mask = np.array(mask)

        if image.shape[2] == 3:
            image = rgb2hsv(image)
            image[:, :, 2] = contrast_stretch(image[:, :, 2])
            image = hsv2rgb(image)
        elif image.shape[2] == 1:
            image = contrast_stretch(image)
        else:
            raise ValueError(
                "Was expecting a 1-channel (greyscale) or 3-channel (colour) image.  Found {} channels"
                .format(image.shape[2]))

        eps = np.finfo(mask.dtype).eps

        coverage = mask.sum(0).sum(0)
        coverage_percent = coverage / coverage.sum()
        coverage_percent = np.clip(coverage_percent, a_min=eps, a_max=1.)

        transformed_image = image
        transformed_mask = mask

        if self.transform is not None:
            while True:
                transformed = self.transform(image=image, mask=mask)
                transformed_image = transformed["image"]
                transformed_mask = transformed["mask"]
                transformed_coverage = transformed_mask.sum(0).sum(0)
                transformed_coverage_percent = transformed_coverage / transformed_coverage.sum(
                )
                transformed_coverage_percent = np.clip(
                    transformed_coverage_percent, a_min=eps, a_max=None)
                relative_coverage = transformed_coverage_percent / coverage_percent
                min_transformed_coverage = relative_coverage[1:].min()

                if min_transformed_coverage > 0.3:
                    break

        return transformed_image, transformed_mask


class BatchDatasetAugmenter(AugmentedDataset):

    lastItem = None

    def __init__(self,
                 dataset,
                 patch_transform,
                 batch_size,
                 augments=None,
                 shuffle=False):
        super().__init__(dataset, patch_transform)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augments = augments

    def __len__(self):
        return len(self.dataset) * self.batch_size

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        dataset_idx = idx // self.batch_size
        flip_idx = idx % self.batch_size

        if flip_idx == 0:
            self.lastItem = super().__getitem__(self.data_map[dataset_idx])

        image, mask = self.lastItem

        if flip_idx == 0:
            pass
        elif flip_idx == 1:
            image = np.flip(image, [0])
            mask = np.flip(mask, [0])
        elif flip_idx == 2:
            image = np.flip(image, [1])
            mask = np.flip(mask, [1])
        elif flip_idx == 3:
            image = np.flip(image, [0, 1])
            mask = np.flip(mask, [0, 1])
        elif self.augments is not None:
            transformed = self.augments(image=image)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            raise ValueError(
                "Batch Size too large, must supply augments to generate additional samples"
            )

        image -= np.min(image)
        image /= np.max(image)
        image = np.clip(image, 0., 1.)
        image = image - np.mean(image)

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())


class DatasetAugmenter(AugmentedDataset):
    def __init__(self,
                 dataset,
                 patch_transform,
                 augments=None,
                 shuffle=False,
                 **kwargs):
        super().__init__(dataset, patch_transform)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle
        self.augments = augments

    def __getitem__(self, idx):

        if idx == 0 and self.shuffle:
            random.shuffle(self.data_map)

        image, mask = super().__getitem__(self.data_map[idx])

        if self.augments is not None:
            image = self.augments(image=image)["image"]

        image -= np.min(image)
        image /= np.max(image)
        image = np.clip(image, 0., 1.)
        image = image - np.mean(image)

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return torch.from_numpy(image), torch.from_numpy(mask)
