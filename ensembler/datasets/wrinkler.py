import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob
from ensembler.datasets.helpers import process_split
import json

num_classes = 4
loss_weights = [0., 1., 1., 1.]
classes = {"background": 0, "gripper": 50, "wrinkle": 100, "fabric": 200}
num_channels = 3


class WrinklerDataset(Dataset):
    """Wrinkler dataset."""
    def __init__(self,
                 wrinkler_folder,
                 train_images=None,
                 val_images=None,
                 test_images=None,
                 split="train"):
        self.split = split
        self.wrinkler_folder = wrinkler_folder

        if self.split == "all":
            file_search = os.path.join(self.wrinkler_folder, "Images", "*.png")
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

    with open(os.path.join(directory, "split.json"), "r") as splitjson:
        sample_split = json.load(splitjson)

    statistics_file = os.path.join(directory, "class_samples.csv")

    train_images, val_images, test_images = process_split(
        sample_split, statistics_file)

    train_data = WrinklerDataset(directory,
                                 train_images,
                                 val_images,
                                 test_images,
                                 split="train")
    val_data = WrinklerDataset(directory,
                               train_images,
                               val_images,
                               test_images,
                               split="val")
    test_data = WrinklerDataset(directory,
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
