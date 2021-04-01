import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from ensembler.datasets.AugmentedDataset import DatasetAugmenter
import pandas as pd
from ensembler.datasets.helpers import split_dataset, sample_dataset
import json

image_height = 768
image_width = 768
num_classes = 4
loss_weights = [1, 1, 2, 1]
classes = {"background": 0, "gripper": 50, "wrinkle": 100, "fabric": 200}
num_channels = 3


class WrinklerDataset(Dataset):
    """Wrinkler dataset."""
    def __init__(self,
                 wrinkler_folder,
                 test_percent=15.,
                 val_percent=5.,
                 split="train"):
        self.split = split
        self.wrinkler_folder = wrinkler_folder

        with open(os.path.join(self.wrinkler_folder, "split.json"),
                  "r") as splitjson:
            sample_split = json.load(splitjson)

        test_images = sample_split["test"]
        trainval_images = sample_split["trainval"]

        statistics_file = os.path.join(self.wrinkler_folder,
                                       "class_samples.csv")
        dataset_df = pd.read_csv(statistics_file)
        trainval_df = dataset_df[dataset_df["sample"].isin(trainval_images)]

        trainval_df = sample_dataset(trainval_df)
        val_df, train_df = split_dataset(trainval_df, 10.)

        val_images = val_df["sample"].tolist()
        train_images = train_df["sample"].tolist()

        if split == "train":
            self.images = train_images
        elif split == "val":
            self.images = val_images
        elif split == "test":
            self.images = test_images
        elif self.split == "all":
            self.images = test_images + trainval_images
        else:
            raise ValueError("Split should be one of train, val, test or all")

        self.images = [
            name for name, ext in [os.path.splitext(i) for i in self.images]
        ]

    def get_image_names(self):
        return self.images

    def load_image(self, image_name):
        image_path = os.path.join(self.wrinkler_folder, "Images",
                                  "{}.png".format(image_name))
        mask_path = os.path.join(self.wrinkler_folder, "Masks1",
                                 "{}.png".format(image_name))
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        one_hot_mask = np.zeros((image.shape[0], image.shape[1], len(classes)),
                                dtype=np.uint8)

        for i, (clazz, value) in enumerate(classes.items()):
            one_hot_mask[:, :, i][mask == value] = 1

        image = image.astype("float32") / 255
        one_hot_mask = one_hot_mask.astype(image.dtype)

        return image_name, image, one_hot_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, image, mask = self.load_image(self.images[idx])
        return (image, mask)


def get_all_dataloader(directory):
    return WrinklerDataset(directory, split="all")


def get_dataloaders(directory, augmenters, batch_size, augmentations):

    train_data = WrinklerDataset(directory, split="train")
    val_data = WrinklerDataset(directory, split="val")
    test_data = WrinklerDataset(directory, split="test")

    train_transform, patch_transform, test_transform = augmentations
    train_augmenter, val_augmenter = augmenters

    train_data = train_augmenter(train_data,
                                 patch_transform,
                                 augments=train_transform,
                                 batch_size=batch_size,
                                 shuffle=True)
    val_data = val_augmenter(val_data, test_transform)
    test_data = val_augmenter(test_data, test_transform)

    return train_data, val_data, test_data
