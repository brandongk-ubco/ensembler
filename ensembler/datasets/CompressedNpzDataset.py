import os
import glob
import numpy as np
import torch
import pandas as pd
from .helpers import process_split, repeat_infrequent_classes
import json
from ensembler.datasets.AugmentedDataset import DatasetAugmenter


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
            self.train_images = train_images
            self.val_images = val_images
            self.test_images = test_images
            if self.split == "train":
                self.images = train_images
            elif self.split == "val":
                self.images = val_images
            elif self.split == "test":
                self.images = val_images + test_images
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

        mask = mask[:, :, 1:]

        mask = mask.astype(image.dtype)

        return image_name, image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, image, mask = self.load_image(self.images[idx])
        return (image, mask)

    @classmethod
    def get_dataloaders(cls, directory, batch_size, augmentations):

        with open(os.path.join(directory, "split.json"), "r") as splitjson:
            sample_split = json.load(splitjson)

        statistics_file = os.path.join(directory, "class_samples.csv")
        statistics = pd.read_csv(statistics_file)

        train_images, val_images, test_images = process_split(
            sample_split, statistics)

        train_images = repeat_infrequent_classes(train_images, statistics)

        train_data = cls(directory,
                         train_images,
                         val_images,
                         test_images,
                         split="train")
        val_data = cls(directory,
                       train_images,
                       val_images,
                       test_images,
                       split="val")
        test_data = cls(directory,
                        train_images,
                        val_images,
                        test_images,
                        split="test")

        preprocessing_transform, train_transform, patch_transform, test_transform = augmentations

        train_data = DatasetAugmenter(
            train_data,
            patch_transform,
            preprocessing_transform=preprocessing_transform,
            augments=train_transform,
            batch_size=batch_size,
            shuffle=True)

        val_data = DatasetAugmenter(
            val_data,
            test_transform,
            preprocessing_transform=preprocessing_transform)

        test_data = DatasetAugmenter(
            test_data,
            test_transform,
            preprocessing_transform=preprocessing_transform,
        )

        return train_data, val_data, test_data

    @classmethod
    def get_all_dataloader(cls, directory):
        return cls(directory, split="all")
