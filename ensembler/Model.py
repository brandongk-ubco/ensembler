import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from ensembler.losses import FocalLoss, SoftBCELoss
from ensembler.utils import crop_image_only_outside


class Segmenter(pl.LightningModule):
    def __init__(self, dataset, train_data, val_data, test_data, **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.patience = self.hparams["patience"]
        self.encoder_name = self.hparams["encoder_name"]
        self.num_workers = self.hparams["num_workers"]
        self.depth = self.hparams["depth"]
        self.batch_size = self.hparams["batch_size"]
        self.batch_loss_multiplier = self.hparams["batch_loss_multiplier"]
        self.focal_loss_multiplier = self.hparams["focal_loss_multiplier"]
        self.bce_loss_multiplier = self.hparams["bce_loss_multiplier"]
        self.dice_loss_multiplier = self.hparams["dice_loss_multiplier"]
        self.weight_decay = self.hparams["weight_decay"]
        self.learning_rate = self.hparams["learning_rate"]
        self.min_learning_rate = self.hparams["min_learning_rate"]
        self.dataset = dataset
        self.val_data = val_data
        self.train_data = train_data
        self.test_data = test_data

        self.train_batches_to_write = 1
        self.val_batches_to_write = 10
        self.intensity = 255 // self.dataset.num_classes

        self.model = self.get_model()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--encoder_name',
                            type=str,
                            default="efficientnet-b3")
        parser.add_argument('--depth', type=int, default=5)
        parser.add_argument('--batch_loss_multiplier',
                            type=float,
                            default=None)
        parser.add_argument('--focal_loss_multiplier', type=float, default=0.)
        parser.add_argument('--dice_loss_multiplier', type=float, default=0.)
        parser.add_argument('--bce_loss_multiplier', type=float, default=1.)
        parser.add_argument('--weight_decay', type=float, default=3e-5)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--min_learning_rate', type=float, default=1e-7)

    def get_model(self):
        model = smp.Unet(encoder_name=self.encoder_name,
                         encoder_weights="imagenet",
                         encoder_depth=self.depth,
                         in_channels=3,
                         classes=self.dataset.num_classes,
                         activation="softmax2d")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(self.dataset.num_channels, 3, (1, 1)),
            torch.nn.BatchNorm2d(self.dataset.num_channels), model)
        return model

    def sample_loss(self, y_hat, y):

        focal_loss = FocalLoss("multilabel",
                               weights=self.dataset.loss_weights,
                               from_logits=False)(y_hat, y)

        dice_loss = smp.losses.DiceLoss(
            "multilabel",
            from_logits=False,
            classes=range(1, self.dataset.num_classes))(y_hat, y)

        bce_loss = SoftBCELoss(from_logits=False,
                               weights=self.dataset.loss_weights)(y_hat, y)

        weighted_bce_loss = self.bce_loss_multiplier * bce_loss
        weighted_focal_loss = self.focal_loss_multiplier * focal_loss
        weighted_dice_loss = self.dice_loss_multiplier * dice_loss

        weighted_loss_values = {
            "bce_loss": weighted_bce_loss,
            "focal_loss": weighted_focal_loss,
            "dice_loss": weighted_dice_loss
        }

        unweighted_loss_values = {
            "bce_loss": bce_loss,
            "focal_loss": focal_loss,
            "dice_loss": dice_loss
        }

        return weighted_loss_values, unweighted_loss_values

    def loss(self, y_hat, y):
        weighted_loss_values, unweighted_loss_values = self.sample_loss(
            y_hat.clone(), y.clone())

        for k, v in unweighted_loss_values.items():
            self.log("unweighted_{}".format(k), v)

        for k, v in weighted_loss_values.items():
            self.log(k, v, prog_bar=True)

        loss = torch.stack(list(weighted_loss_values.values())).sum()

        return loss

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.write_predictions(x,
                               y,
                               y_hat,
                               batch_idx,
                               prefix="train",
                               batches_to_write=self.train_batches_to_write)

        return loss

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.write_predictions(x,
                               y,
                               y_hat,
                               batch_idx,
                               prefix="val",
                               batches_to_write=self.val_batches_to_write)

        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss", loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            min_lr=self.min_learning_rate,
            verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }

    def save_prediction(self, img, mask_img, predicted_mask_img, outfile):

        fig, axs = plt.subplots(3, 1)

        if img.shape[2] == 1:
            axs[0].imshow(img.squeeze(), cmap="gray")
        else:
            axs[0].imshow(img)

        axs[1].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
        axs[2].imshow(predicted_mask_img, cmap="gray", vmin=0, vmax=255)

        for ax_i in axs:
            ax_i.axis('off')

        if os.path.exists(outfile):
            os.remove(outfile)

        plt.savefig(outfile)
        plt.close()

    def write_predictions(self,
                          x,
                          y,
                          y_hat,
                          batch_idx,
                          prefix="",
                          batches_to_write=1):
        x = x.clone().detach().cpu().numpy()
        y = y.clone().detach().cpu().numpy()
        y_hat = y_hat.clone().detach().cpu().numpy()

        if batch_idx >= batches_to_write:
            return

        try:
            self.logger.log_dir
        except AttributeError:
            return

        for i in range(y_hat.shape[0]):
            img = x[i, :, :, :]
            #img = img - np.min(img)
            img = img.transpose(1, 2, 0)

            mask = y[i, :, :, :]
            mask = mask.transpose(1, 2, 0)
            mask_img = np.argmax(mask, axis=2) * self.intensity

            predicted_mask = y_hat[i, :, :, :]
            predicted_mask = predicted_mask.transpose(1, 2, 0)
            predicted_mask_img = np.argmax(predicted_mask,
                                           axis=2) * self.intensity

            row_start, row_end, col_start, col_end = crop_image_only_outside(
                img)

            img = img[row_start:row_end, col_start:col_end, :]
            mask_img = mask_img[row_start:row_end, col_start:col_end]
            predicted_mask_img = predicted_mask_img[row_start:row_end,
                                                    col_start:col_end]

            outfile = os.path.join(self.logger.log_dir,
                                   "{}_{}_{}.png".format(prefix, batch_idx, i))

            self.save_prediction(img, mask_img, predicted_mask_img, outfile)
