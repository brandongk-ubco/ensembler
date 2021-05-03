import os
import json
from ensembler.datasets.helpers import process_split, repeat_infrequent_classes
import pandas as pd


def base_get_dataloaders(directory, augmenters, batch_size, augmentations,
                         Dataset):

    with open(os.path.join(directory, "split.json"), "r") as splitjson:
        sample_split = json.load(splitjson)

    statistics_file = os.path.join(directory, "class_samples.csv")
    statistics = pd.read_csv(statistics_file)

    train_images, val_images, test_images = process_split(
        sample_split, statistics)

    train_images = repeat_infrequent_classes(train_images, statistics)

    train_data = Dataset(directory,
                         train_images,
                         val_images,
                         test_images,
                         split="train")
    val_data = Dataset(directory,
                       train_images,
                       val_images,
                       test_images,
                       split="val")
    test_data = Dataset(directory,
                        train_images,
                        val_images,
                        test_images,
                        split="test")

    preprocessing_transform, train_transform, patch_transform, test_transform = augmentations
    train_augmenter, val_augmenter, test_augmenter = augmenters

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

    test_data = test_augmenter(
        test_data,
        test_transform,
        preprocessing_transform=preprocessing_transform,
    )

    return train_data, val_data, test_data


def base_get_all_dataloader(directory, Dataset):
    return Dataset(directory, split="all")
