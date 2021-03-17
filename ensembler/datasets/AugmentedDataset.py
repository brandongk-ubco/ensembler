import numpy as np
import torch


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

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        return image, mask


class DatasetAugmenter(AugmentedDataset):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        return image, mask
