import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from ensembler.datasets._base import base_get_dataloaders, base_get_all_dataloader
from functools import partial

num_classes = 5
loss_weights = [0., 1., 1., 1., 1.]
classes = {"background": 0, "1": 50, "2": 100, "3": 200, "4": 250}
num_channels = 1


class SeverstalDataset(Dataset):
    """Severstal dataset."""

    cache = {}

    def __init__(self,
                 severstal_folder,
                 train_images=None,
                 val_images=None,
                 test_images=None,
                 split="train"):
        self.split = split
        self.severstal_folder = severstal_folder

        if self.split == "all":
            file_search = os.path.join(self.severstal_folder, "*.npz")
            files = glob.glob(file_search)
            self.images = [
                os.path.splitext(os.path.basename(f))[0] for f in files
            ]
        else:
            assert train_images is not None
            assert val_images is not None
            assert test_images is not None
            if self.split == "train":
                self.images = train_images
            elif self.split == "val":
                self.images = val_images
            elif self.split == "test":
                self.images = test_images
            else:
                raise ValueError(
                    "Split should be one of train, val, test or all")

    def get_image_names(self):
        return self.images

    def load_image(self, image_name):
        image_file = os.path.join(self.severstal_folder,
                                  "{}.npz".format(image_name))
        image_np = np.load(image_file)

        image = image_np["image"]
        mask = image_np["mask"]

        background = np.expand_dims((np.sum(mask,
                                            axis=2) == 0).astype(mask.dtype),
                                    axis=2)

        mask = np.concatenate([background, mask], axis=2)

        image = np.expand_dims(image, axis=2)

        return image_name, image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, image, mask = self.load_image(self.images[idx])
        return (image, mask)


get_dataloaders = partial(base_get_dataloaders, Dataset=SeverstalDataset)
get_all_dataloader = partial(base_get_all_dataloader, Dataset=SeverstalDataset)
