import glob
import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from datasets.AugmentedDataset import DatasetAugmenter
import pandas as pd
from utils import split_dataframe

image_height = 256
image_width = 1600
batch_size = 8
num_classes = 5
loss_weights = [1.063732, 697.93036, 3272.005379, 20.793984, 99.165978]
classes = {"background": 0, "1": 50, "2": 100, "3": 200, "4": 250}


class SeverstalDataset(Dataset):
    """Severstal dataset."""

    cache = {}

    def __init__(self, severstal_folder, val_percent=1., split="train"):
        self.split = split
        self.severstal_folder = severstal_folder

        images = [
            os.path.basename(i)
            for i in glob.glob(os.path.join(self.severstal_folder, "*.npz"))
        ]

        with open(os.path.join(self.severstal_folder, "split.json"),
                  "r") as splitjson:
            sample_split = json.load(splitjson)

        test_images = sample_split["test"]
        trainval_images = sample_split["trainval"]

        statistics_file = os.path.join(self.severstal_folder,
                                       "class_samples.csv")
        dataset_df = pd.read_csv(statistics_file)
        trainval_df = dataset_df[dataset_df["sample"].isin(trainval_images)]

        val_df, train_df = split_dataframe(trainval_df, 10.)

        val_images = val_df["sample"].tolist()
        train_images = train_df["sample"].tolist()

        assert len(test_images) + len(val_images) + len(train_images) == len(
            images)
        if self.split == "train":
            self.images = train_images
        elif self.split == "val":
            self.images = val_images
        elif self.split == "test":
            self.images = test_images
        elif self.split == "all":
            self.images = test_images + trainval_images
        else:
            raise ValueError("Split should be one of train, val, test or all")

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


def get_dataloaders(directory, augmentations):

    train_data = SeverstalDataset(directory, split="train")
    val_data = SeverstalDataset(directory, split="val")
    test_data = SeverstalDataset(directory, split="test")
    all_data = SeverstalDataset(directory, split="all")

    assert len(
        set(train_data.images + val_data.images +
            test_data.images)) == len(train_data.images + val_data.images +
                                      test_data.images)

    assert len(train_data.images + val_data.images + test_data.images) == len(
        all_data.images)

    assert len(
        set(train_data.images + val_data.images + test_data.images +
            all_data.images)) == len(all_data.images)

    train_transform, val_transform, test_transform = augmentations
    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data, all_data
