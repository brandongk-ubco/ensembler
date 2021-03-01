import glob
import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np
from datasets.AugmentedDataset import DatasetAugmenter

image_height = 1792
image_width = 2048
batch_size = 2
num_classes = 4
loss_weights = [0.5, 1, 10, 1]


class WrinklerDataset(Dataset):
    """Wrinkler dataset."""

    cache = {}

    classes = {"background": 0, "gripper": 50, "wrinkle": 100, "fabric": 200}

    def __init__(self,
                 wrinkler_folder,
                 test_percent=15.,
                 val_percent=5.,
                 split="train"):
        self.split = split
        self.wrinkler_folder = wrinkler_folder

        images = [
            os.path.basename(i) for i in glob.glob(
                os.path.join(self.wrinkler_folder, "Images", "*.png"))
        ]

        randomizer = random.Random(42)
        randomizer.shuffle(images)
        num_test_images = round(len(images) * test_percent / 100.)
        num_trainval_images = len(images) - num_test_images
        num_val_images = round(num_trainval_images * val_percent / 100.)

        test_images = images[:num_test_images]
        val_images = images[num_test_images:num_test_images + num_val_images]
        train_images = images[num_test_images + num_val_images:]

        assert len(test_images) + len(val_images) + len(train_images) == len(
            images)
        if split == "train":
            self.images = train_images
        elif split == "val":
            self.images = val_images
        elif split == "test":
            self.images = test_images

    def load_image(self, image_name):
        image_path = os.path.join(self.wrinkler_folder, "Images", image_name)
        mask_path = os.path.join(self.wrinkler_folder, "Masks1", image_name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        one_hot_mask = np.zeros(
            (image.shape[0], image.shape[1], len(self.classes)),
            dtype=np.uint8)

        for i, (clazz, value) in enumerate(self.classes.items()):
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


def get_dataloaders(directory, augmentations):
    train_transform, val_transform, test_transform = augmentations

    train_data = WrinklerDataset(directory, split="train")
    val_data = WrinklerDataset(directory, split="val")
    test_data = WrinklerDataset(directory, split="test")

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data
