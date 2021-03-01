import torchvision
import numpy as np
import torch
from datasets.AugmentedDataset import DatasetAugmenter
from torch.utils.data import Dataset
from PIL import Image
from p_tqdm import p_umap
import os

image_height = 512
image_width = 512
num_classes = 21
batch_size = 8

voc_folder = "/mnt/d/work/datasets/voc"


class VOCDataset(Dataset):

    cache = {"images": {}, "masks": {}}

    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "potted plant", "sheep", "sofa", "train",
        "tv/monitor"
    ]

    def __init__(self, voc_folder, val_percent=5., split="train"):

        self.split = split

        # This takes lots of memory... too much for 32GB
        self.use_cache = False

        if split == "train":
            self.data = torchvision.datasets.VOCSegmentation(voc_folder,
                                                             image_set='train')
        elif split == "val":
            self.data = torchvision.datasets.VOCSegmentation(voc_folder,
                                                             image_set='val')
        elif split == "test":
            raise ValueError("Test split not implemented.")
        else:
            raise ValueError("Incorrect split {}".format(split))

        self.samples = [i for i in zip(self.data.images, self.data.masks)]

        if self.use_cache:
            self.populate_cache()

    def get_image_names(self):
        return [
            os.path.splitext(os.path.basename(i))[0] for i in self.data.images
        ]

    def load_image(self, sample):
        image_path, mask_path = sample

        if self.use_cache and image_path in self.cache["images"]:
            image = self.cache["images"][image_path]
        else:
            image = Image.open(image_path)
            image = np.array(image)
            image = image.astype("float32") / 255.
            image = torch.Tensor(image)
            if self.use_cache:
                self.cache["images"][image_path] = image

        if self.use_cache and mask_path in self.cache["masks"]:
            label_mask = self.cache["masks"][mask_path]
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask[mask == 255] = 0
            label_mask = np.zeros(
                (len(self.classes), mask.shape[0], mask.shape[1]),
                dtype=np.float32)

            for k, v in enumerate(self.classes):
                label_mask[k, mask == k] = 1

            label_mask = np.round(label_mask, 0)
            label_mask = label_mask.transpose(1, 2, 0)
            label_mask = torch.Tensor(label_mask)

            if self.use_cache:
                self.cache["masks"][mask_path] = label_mask

        return image, label_mask

    def populate_cache(self):
        print("Populating image cache for {}.".format(self.split))
        p_umap(self.load_image, self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.load_image(self.samples[idx])


def get_dataloaders(augmentations):
    train_transform, val_transform, test_transform = augmentations

    train_data = VOCDataset(voc_folder, split="train")
    val_data = VOCDataset(voc_folder, split="val")
    test_data = VOCDataset(voc_folder, split="val")

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data
