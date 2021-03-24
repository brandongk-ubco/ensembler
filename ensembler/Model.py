import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datasets import Datasets
from utils import weighted_loss, ds_combination
from functools import partial
from typing import List


def ds_reduce_column(x: int, y: int, y_hat, reduced) -> torch.Tensor:
    for k in range(1, y_hat.shape[0]):
        reduced[:, x, y] = ds_combination(reduced[:, x, y], y_hat[k, :, x, y])
    return torch.empty(0)


@torch.jit.script
def ds_reduce(y_hat):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    reduced = y_hat[0, :, :, :].clone()
    expected = y_hat.shape[2] * y_hat.shape[3]

    i = 0

    for x in range(y_hat.shape[2]):
        for y in range(y_hat.shape[3]):
            if len(futures) > 100:
                for x, future in enumerate(futures):
                    if i % 10000 == 0:
                        print(i / expected)
                    i += 1
                    torch.jit.wait(future)
                futures.clear()
            futures.append(
                torch.jit.fork(ds_reduce_column, x, y, y_hat, reduced))

    return reduced


def batch_loss(y_hat,
               y,
               reduction=partial(torch.mean, **{"dim": [0]}),
               loss=smp.losses.FocalLoss("multilabel")):

    batches = y_hat.shape[0]
    loss_val = 0
    for batch in range(0, batches, 4):
        y_hat_batch = y_hat[batch:batch + 4, :, :, :].clone()
        y_hat_batch[1, :, :, :] = torch.flip(y_hat_batch[1, :, :, :].clone(),
                                             [1])
        y_hat_batch[2, :, :, :] = torch.flip(y_hat_batch[2, :, :, :].clone(),
                                             [2])
        y_hat_batch[3, :, :, :] = torch.flip(y_hat_batch[3, :, :, :].clone(),
                                             [1, 2])
        y_batch = y[batch:batch + 4, :, :, :].clone()
        y_batch[1, :, :, :] = torch.flip(y_batch[1, :, :, :].clone(), [1])
        y_batch[2, :, :, :] = torch.flip(y_batch[2, :, :, :].clone(), [2])
        y_batch[3, :, :, :] = torch.flip(y_batch[3, :, :, :].clone(), [1, 2])

        try:
            eps = torch.finfo(y.dtype).eps
            assert torch.all(y_batch[0, :, :, :] - y_batch[1, :, :, :] <= eps)
            assert torch.all(y_batch[0, :, :, :] - y_batch[2, :, :, :] <= eps)
            assert torch.all(y_batch[0, :, :, :] - y_batch[3, :, :, :] <= eps)
            assert torch.all(y_batch[1, :, :, :] - y_batch[0, :, :, :] <= eps)
            assert torch.all(y_batch[2, :, :, :] - y_batch[0, :, :, :] <= eps)
            assert torch.all(y_batch[3, :, :, :] - y_batch[0, :, :, :] <= eps)
        except AssertionError:
            print(torch.sum(y_batch[0, :, :, :] - y_batch[1, :, :, :]))
            print(torch.sum(y_batch[0, :, :, :] - y_batch[2, :, :, :]))
            print(torch.sum(y_batch[0, :, :, :] - y_batch[3, :, :, :]))

        y_hat_batch = reduction(y_hat_batch)
        y_batch = y_batch[0, :, :, :]

        loss_val += loss(y_hat_batch, y_batch)

    return loss_val


class Segmenter(pl.LightningModule):
    def __init__(self, get_augments, **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.patience = self.hparams["patience"]
        self.dataset = Datasets.get(self.hparams["dataset"])
        self.train_data, self.val_data, self.test_data, self.all_data = self.dataset.get_dataloaders(
            os.path.join(self.hparams["data_dir"], self.hparams["dataset"]),
            get_augments(self.dataset.image_height, self.dataset.image_width))
        self.encoder_name = self.hparams["encoder_name"]
        self.num_workers = self.hparams["num_workers"]
        self.depth = self.hparams["depth"]
        self.batch_size = self.hparams["batch_size"]

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.batches_to_write = 2
        self.intensity = 255 // self.dataset.num_classes

        self.model = self.get_model()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_name',
                            type=str,
                            default="efficientnet-b0")
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--num_workers',
                            type=int,
                            default=os.cpu_count() // 2)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--depth', type=int, default=5)
        parser.add_argument('--data_dir',
                            type=str,
                            nargs='?',
                            const=os.environ.get("DATA_DIR", None),
                            default=os.environ.get("DATA_DIR", None))
        return parser

    def get_model(self):
        model = smp.Unet(encoder_name=self.encoder_name,
                         encoder_weights=None,
                         encoder_depth=self.depth,
                         in_channels=self.dataset.num_channels,
                         classes=self.dataset.num_classes,
                         activation='softmax2d')
        model.apply(self.initialize_weights)
        return model

    def initialize_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight.data, 1)
            torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0)

    def get_loss(self):
        focal_kwargs = {
            "weights": self.dataset.loss_weights,
            "loss_function": smp.losses.FocalLoss("binary", reduction=None)
        }
        focal_loss = partial(weighted_loss, **focal_kwargs)

        # minimize_diversity_kwargs = {
        #     "reduction": partial(torch.std, **{"dim": [0]}),
        #     "loss": lambda y_hat, y: torch.mean(y_hat)
        # }
        # minimize_diversity = partial(batch_loss, **minimize_diversity_kwargs)

        return lambda y_hat, y: focal_loss(y_hat, y) + batch_loss(y_hat, y)

    def get_optimizer(self):
        return torch.optim.Adam

    def all_dataloader(self):
        return torch.utils.data.DataLoader(self.all_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        assert y_hat.shape == y.shape
        assert x.shape[0] == y.shape[0]
        assert x.shape[2:] == y.shape[2:]

        loss = self.loss(y_hat, y)
        return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.write_predictions(x.clone().detach().cpu().numpy(),
                               y.clone().detach().cpu().numpy(),
                               y_hat.clone().detach().cpu().numpy(), batch_idx)
        return {"val_loss", loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, min_lr=1e-5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }

    def write_predictions(self, x, y, y_hat, batch_idx):

        if batch_idx >= self.batches_to_write:
            return

        try:
            self.logger.log_dir
        except AttributeError:
            return

        for i in range(y_hat.shape[0]):
            img = x[i, :, :, :]
            mask = y[i, :, :, :]
            predicted_mask = y_hat[i, :, :, :]

            img = img.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)
            predicted_mask = predicted_mask.transpose(1, 2, 0)

            mask_img = np.argmax(mask, axis=2) * self.intensity

            predicted_mask_img = np.argmax(predicted_mask,
                                           axis=2) * self.intensity

            fig, axs = plt.subplots(3, 1)

            if img.shape[2] == 1:
                axs[0].imshow(img.squeeze(), cmap="gray")
            else:
                axs[0].imshow(img, cmap="gray")

            axs[1].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
            axs[2].imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)

            for ax_i in axs:
                ax_i.axis('off')

            outfile = os.path.join(self.logger.log_dir,
                                   "{}_{}.png".format(batch_idx, i))

            if os.path.exists(outfile):
                os.remove(outfile)

            plt.savefig(outfile)
            plt.close()
