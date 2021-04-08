import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
import pandas as pd
from ensembler.datasets.helpers import process_split
import glob

image_height = 256
image_width = 256
num_classes = 5
# loss_weights = [1.063732, 697.93036, 3272.005379, 20.793984, 99.165978]
loss_weights = [1., 1., 1., 1., 1.]
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


def get_all_dataloader(directory):
    return SeverstalDataset(directory, split="all")


def get_dataloaders(directory, augmenters, batch_size, augmentations):

    with open(os.path.join(directory, "split.json"), "r") as splitjson:
        sample_split = json.load(splitjson)

    statistics_file = os.path.join(directory, "class_samples.csv")

    train_images, val_images, test_images = process_split(
        sample_split, statistics_file)

    train_data = SeverstalDataset(directory,
                                  train_images,
                                  val_images,
                                  test_images,
                                  split="train")
    val_data = SeverstalDataset(directory,
                                train_images,
                                val_images,
                                test_images,
                                split="val")
    test_data = SeverstalDataset(directory,
                                 train_images,
                                 val_images,
                                 test_images,
                                 split="test")

    preprocessing_transform, train_transform, patch_transform, test_transform = augmentations
    train_augmenter, val_augmenter = augmenters

    train_data = train_augmenter(
        train_data,
        patch_transform,
        preprocessing_transform=preprocessing_transform,
        augments=train_transform,
        batch_size=batch_size,
        shuffle=True)
    val_data = val_augmenter(val_data,
                             test_transform,
                             preprocessing_transform=preprocessing_transform)
    test_data = val_augmenter(
        test_data,
        test_transform,
        preprocessing_transform=preprocessing_transform,
    )

    return train_data, val_data, test_data
