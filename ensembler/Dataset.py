from pytorch_lightning.core.datamodule import LightningDataModule
import torch
import os
from ensembler.datasets import Datasets
from ensembler.augments import get_augments


class Dataset(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 dataset: Datasets,
                 num_workers: int = os.environ.get("NUM_WORKERS",
                                                   os.cpu_count() - 1),
                 batch_size: int = 10,
                 dataset_split_seed: int = 42,
                 patch_height: int = 512,
                 patch_width=512):
        super().__init__()
        self.dataset = Datasets.get(dataset.value)
        self.data_dir = os.path.join(os.path.abspath(data_dir), dataset.value)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_split_seed = dataset_split_seed
        self.patch_height = patch_height
        self.patch_width = patch_width

    def prepare_data(self):
        self.augments = get_augments(self.patch_height, self.patch_width)
        self.train_data, self.val_data, self.test_data = self.dataset.get_dataloaders(
            self.data_dir, self.batch_size, self.augments)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=1,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=1,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False)
