import numpy as np
import torch
from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure
from skimage.util import dtype_limits
import torchvision
import random


def contrast_stretch(image, min_percentile=2, max_percentile=98):
    p2, p98 = np.percentile(image, (min_percentile, max_percentile))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    min_val, max_val = dtype_limits(image, clip_negative=True)
    image = np.clip(image, min_val, max_val)
    return image


class AugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * 4

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

        image -= np.min(image)
        image /= np.max(image)
        image = np.clip(image, 0., 1.)

        eps = np.finfo(mask.dtype).eps

        coverage = mask.sum(0).sum(0)
        coverage_percent = coverage / coverage.sum()
        coverage_percent = np.clip(coverage_percent, a_min=eps, a_max=1.)
        min_transformed_coverage = 0.

        while min_transformed_coverage < 0.5:
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

        transformed_image = transformed_image.transpose(2, 0, 1)
        transformed_mask = transformed_mask.transpose(2, 0, 1)

        return torch.from_numpy(transformed_image), torch.from_numpy(
            transformed_mask)


class DatasetAugmenter(AugmentedDataset):

    lastItem = None

    def __init__(self, dataset, transform, shuffle=False):
        super().__init__(dataset, transform)
        num_elements = len(self.dataset)
        self.data_map = list(range(num_elements))
        self.shuffle = shuffle

    def __getitem__(self, idx):
        if idx == 0:
            random.shuffle(self.data_map)

        dataset_idx = idx // 4
        flip_idx = idx % 4

        if flip_idx == 0:
            self.lastItem = super().__getitem__(self.data_map[dataset_idx])

        image, mask = self.lastItem

        if flip_idx == 0:
            pass

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
