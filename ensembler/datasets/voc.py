import torchvision
from .AugmentedDataset import AugmentedDataset

dataset_path = "/mnt/e/datasets/voc/"

train_data = torchvision.datasets.VOCSegmentation(dataset_path,
                                                  image_set='train')

val_data = torchvision.datasets.VOCSegmentation(dataset_path,
                                                image_set='trainval')


class VOCAugmentedDataset(AugmentedDataset):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        mask[mask > 0] = 1.

        return image, mask
