import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from ensembler.datasets.AugmentedDataset import DatasetAugmenter

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

image_height = 512
image_width = 512
num_classes = len(classes)
batch_size = 8
loss_weights = [1.] * num_classes


class VOCDataset(Dataset):
    def __init__(self, voc_folder, val_percent=5., split="train"):

        self.split = split

        if split == "train":
            self.data = torchvision.datasets.VOCSegmentation(
                voc_folder,
                image_set='train',
                download=not os.path.exists(voc_folder))
        elif split == "val":
            self.data = torchvision.datasets.VOCSegmentation(
                voc_folder,
                image_set='val',
                download=not os.path.exists(voc_folder))
        elif self.split == "all":
            self.images = self.data = torchvision.datasets.VOCSegmentation(
                voc_folder,
                image_set='trainval',
                download=not os.path.exists(voc_folder))
        elif split == "test":
            raise ValueError("Test split not implemented.")
        else:
            raise ValueError("Incorrect split {}".format(split))

        self.samples = [i for i in zip(self.data.images, self.data.masks)]

    def get_image_names(self):
        return [
            os.path.splitext(os.path.basename(i))[0] for i in self.data.images
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


def get_dataloaders(voc_folder, augmentations):

    train_data = VOCDataset(voc_folder, split="train")
    val_data = VOCDataset(voc_folder, split="val")
    test_data = VOCDataset(voc_folder, split="val")
    all_data = VOCDataset(voc_folder, split="all")

    train_transform, val_transform, test_transform = augmentations
    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data, all_data
