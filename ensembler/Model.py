import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from datasets import Datasets
from utils import weighted_loss
from functools import partial


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
        return smp.Unet(encoder_name=self.encoder_name,
                        encoder_weights=None,
                        encoder_depth=self.depth,
                        in_channels=self.dataset.num_channels,
                        classes=self.dataset.num_classes,
                        activation='softmax2d')

    def get_loss(self):
        kwargs = {
            "weights": self.dataset.loss_weights,
            "loss_function": smp.losses.FocalLoss("multilabel", reduction=None)
        }
        focal_loss = partial(weighted_loss, **kwargs)

        return lambda y_hat, y: focal_loss(y_hat, y) + smp.losses.DiceLoss(
            "multilabel")(y_hat, y)

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
                                           shuffle=True,
                                           drop_last=True)

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

        for i in range(x.shape[0]):
            img = x[i, :, :, :]
            mask = y[i, :, :, :]
            predicted_mask = y_hat[i, :, :, :]

            img = img.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)
            predicted_mask = predicted_mask.transpose(1, 2, 0)

            mask_img = np.argmax(mask, axis=2) * self.intensity

            predicted_mask_img = np.argmax(predicted_mask,
                                           axis=2) * self.intensity

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

            if img.shape[2] == 1:
                ax1.imshow(img.squeeze(), cmap="gray")
            else:
                ax1.imshow(img, cmap="gray")

            ax2.imshow(mask_img, cmap="gray", vmin=0, vmax=255)
            ax3.imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

            outfile = os.path.join(self.logger.log_dir,
                                   "{}_{}.png".format(batch_idx, i))

            if os.path.exists(outfile):
                os.remove(outfile)

            plt.savefig(outfile)
            plt.close()
