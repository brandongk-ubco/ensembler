import glob
import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from p_tqdm import t_map as mapper
from datasets.AugmentedDataset import DatasetAugmenter

image_height = 256
image_width = 1600
batch_size = 4
num_classes = 5


class SeverstalDataset(Dataset):
    """Severstal dataset."""

    cache = {}

    classes = {"background": 0, "1": 50, "2": 100, "3": 200, "4": 250}

    def __init__(self,
                 severstal_folder,
                 test_percent=15.,
                 val_percent=5.,
                 split="train",
                 use_cache=True):
        self.split = split
        self.use_cache = use_cache
        self.severstal_folder = severstal_folder

        images = [
            os.path.basename(i)
            for i in glob.glob(os.path.join(self.severstal_folder, "*.npz"))
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

        if self.use_cache:
            self.populate_cache()

    def load_image(self, image_name):
        image_file = os.path.join(self.severstal_folder, image_name)
        image_np = np.load(image_file)

        image = image_np["image"]
        mask = image_np["mask"]

        background = np.expand_dims((np.sum(mask,
                                            axis=2) == 0).astype(mask.dtype),
                                    axis=2)

        mask = np.concatenate([background, mask], axis=2)

        image = np.expand_dims(image, axis=2)

        return image_name, image, mask

    def populate_cache(self):
        print("Populating image cache for {}.".format(self.split))
        for image_name, image, mask in mapper(self.load_image, self.images):
            self.cache[image_name] = (image, mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.use_cache:
            return self.cache[self.images[idx]]
        else:
            image_name, image, mask = self.load_image(self.images[idx])
            return (image, mask)


def get_dataloaders(augmentations, use_cache=False):
    train_transform, val_transform, test_transform = augmentations

    train_data = SeverstalDataset("/mnt/d/work/datasets/severstal/",
                                  split="train",
                                  use_cache=use_cache)
    val_data = SeverstalDataset("/mnt/d/work/datasets/severstal/",
                                split="val",
                                use_cache=use_cache)
    test_data = SeverstalDataset("/mnt/d/work/datasets/severstal/",
                                 split="test",
                                 use_cache=use_cache)

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data
