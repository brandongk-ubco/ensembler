import torchvision
import numpy as np
import torch
from datasets.AugmentedDataset import DatasetAugmenter
from torch.utils.data import Dataset

image_height = 512
image_width = 512
num_classes = 21
batch_size = 20

voc_folder = "/mnt/d/work/datasets/voc"


class VOCDataset(Dataset):

    cache = {}

    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "potted plant", "sheep", "sofa", "train",
        "tv/monitor"
    ]

    def __init__(self, voc_folder, val_percent=5., split="train"):

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

    def load_image(self, sample):
        image, mask = sample

        image = np.array(image)
        mask = np.array(mask)

        mask[mask == 255] = 0

        label_mask = np.zeros(
            (len(self.classes), mask.shape[0], mask.shape[1]),
            dtype=image.dtype)

        for k, v in enumerate(self.classes):
            label_mask[k, mask == k] = 1

        image = image.astype("float32") / 255.
        label_mask = label_mask.transpose(1, 2, 0)

        label_mask = torch.Tensor(label_mask)
        image = torch.Tensor(image)

        return image, label_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.load_image(self.data[idx])


def get_dataloaders(augmentations):
    train_transform, val_transform, test_transform = augmentations

    train_data = VOCDataset(voc_folder, split="train")
    val_data = VOCDataset(voc_folder, split="val")
    test_data = VOCDataset(voc_folder, split="val")

    train_data = DatasetAugmenter(train_data, train_transform)
    val_data = DatasetAugmenter(val_data, val_transform)
    test_data = DatasetAugmenter(test_data, test_transform)

    return train_data, val_data, test_data
