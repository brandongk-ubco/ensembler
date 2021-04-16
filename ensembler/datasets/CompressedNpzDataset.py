import os
import glob
import numpy as np
import torch
from ensembler.datasets._base import base_get_dataloaders, base_get_all_dataloader
from functools import partial


class CompressedNpzDataset:
    def __init__(self,
                 folder,
                 train_images=None,
                 val_images=None,
                 test_images=None,
                 split="train"):

        self.split = split
        self.folder = folder

        if self.split == "all":
            file_search = os.path.join(self.folder, "*.npz")
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
        image_file = os.path.join(self.folder, "{}.npz".format(image_name))
        image_np = np.load(image_file)

        image = image_np["image"]
        mask = image_np["mask"]

        return image_name, image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, image, mask = self.load_image(self.images[idx])
        return (image, mask)
