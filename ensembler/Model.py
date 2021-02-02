import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt


class Segmenter(pl.LightningModule):
    def __init__(self, dataset, get_augments, patience=10):
        super().__init__()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.patience = 10
        self.batches_to_write = 2
        self.num_classes = dataset.num_classes
        self.train_data, self.val_data, self.test_data = dataset.get_dataloaders(
            get_augments(dataset.image_height, dataset.image_width))
        self.batch_size = dataset.batch_size

        self.model = self.get_model()
        self.intensity = 255 // self.num_classes

    def get_model(self):
        return smp.Unet(encoder_name="efficientnet-b0",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                        activation='softmax2d')

    def get_loss(self):
        return lambda y_hat, y: smp.losses.DiceLoss("multilabel")(
            y_hat, y) + smp.losses.FocalLoss("multilabel")(y_hat, y)

    def get_optimizer(self):
        return torch.optim.Adam

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.write_predictions(x, y, y_hat, batch_idx)
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

        imgs = x.clone().detach().cpu()
        predicted_masks = y_hat.clone().detach().cpu()
        masks = y.clone().detach().cpu()
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].numpy()
            mask = masks[i, :, :, :].numpy()
            predicted_mask = predicted_masks[i, :, :, :].numpy()

            mask_img = np.argmax(mask, axis=0) * self.intensity
            predicted_mask_img = np.argmax(predicted_mask,
                                           axis=0) * self.intensity

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

            ax1.imshow(img.swapaxes(0, 2).swapaxes(0, 1), vmin=0, vmax=1)
            ax2.imshow(mask_img, cmap="gray", vmin=0, vmax=255)
            ax3.imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

            plt.savefig(
                os.path.join(self.logger.log_dir,
                             "{}_{}.png".format(batch_idx, i)))
            plt.close()
