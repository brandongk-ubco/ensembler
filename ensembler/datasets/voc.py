import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from ensembler.datasets.helpers import process_split
import random

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

num_classes = len(classes)
loss_weights = [1.] * num_classes
loss_weights[0] = 0.
num_channels = 3


class VOCDataset(Dataset):
    def __init__(self,
                 voc_folder,
                 train_images=None,
                 val_images=None,
                 test_images=None,
                 split="train"):

        self.split = split

        data = torchvision.datasets.VOCSegmentation(
            voc_folder,
            image_set='trainval',
            download=not os.path.exists(voc_folder))

        if self.split == "all":
            self.samples = [i for i in zip(data.images, data.masks)]
        else:
            assert train_images is not None
            assert val_images is not None
            assert test_images is not None
            if split == "train":
                sample_from = train_images
            elif split == "val":
                sample_from = val_images
            elif split == "test":
                sample_from = test_images
            else:
                raise ValueError("Incorrect split {}".format(split))

            self.samples = [
                i for i in zip(data.images, data.masks)
                if os.path.splitext(os.path.basename(i[0]))[0] in sample_from
            ]

            r = random.Random()
            r.seed(42)
            r.shuffle(self.samples)

    def get_image_names(self):
        return [
            os.path.splitext(os.path.basename(i))[0] for i, m in self.samples
        ]

    def load_image(self, sample):
        image_path, mask_path = sample

        image = Image.open(image_path)
        image = np.array(image)
        image = image.astype("float32") / 255.
        image = torch.Tensor(image)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask[mask == 255] = 0
        label_mask = np.zeros((num_classes, mask.shape[0], mask.shape[1]),
                              dtype=np.float32)

        for k, v in enumerate(classes):
            label_mask[k, mask == k] = 1

        label_mask = np.round(label_mask, 0)
        label_mask = label_mask.transpose(1, 2, 0)
        label_mask = torch.Tensor(label_mask)

        return image, label_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.load_image(self.samples[idx])


def get_all_dataloader(voc_folder):
    return VOCDataset(voc_folder, split="all")


def get_dataloaders(voc_folder, augmenters, batch_size, augmentations):

    with open(os.path.join(voc_folder, "split.json"), "r") as splitjson:
        sample_split = json.load(splitjson)

    statistics_file = os.path.join(voc_folder, "class_samples.csv")

    train_images, val_images, test_images = process_split(
        sample_split, statistics_file)

    train_data = VOCDataset(voc_folder,
                            train_images,
                            val_images,
                            test_images,
                            split="train")
    val_data = VOCDataset(voc_folder,
                          train_images,
                          val_images,
                          test_images,
                          split="val")
    test_data = VOCDataset(voc_folder,
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
