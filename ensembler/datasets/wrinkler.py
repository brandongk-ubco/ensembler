import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob
from ensembler.datasets._base import base_get_dataloaders, base_get_all_dataloader
from functools import partial

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


get_dataloaders = partial(base_get_dataloaders, Dataset=WrinklerDataset)
get_all_dataloader = partial(base_get_all_dataloader, Dataset=WrinklerDataset)