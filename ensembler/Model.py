import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from ensembler.losses import FocalLoss, SoftBCELoss
from ensembler.utils import crop_image_only_outside
from ensembler.aggregators import harmonize_batch


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
        self.train_batches_to_write = self.hparams["train_batches_to_write"]
        self.val_batches_to_write = self.hparams["val_batches_to_write"]
        self.dataset = dataset
        self.val_data = val_data
        self.train_data = train_data
        self.test_data = test_data

        self.intensity = 255 // self.dataset.num_classes

        self.model = self.get_model()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--encoder_name',
                            type=str,
                            default="efficientnet-b3")
        parser.add_argument('--depth', type=int, default=5)
        parser.add_argument('--focal_loss_multiplier', type=float, default=0.)
        parser.add_argument('--dice_loss_multiplier', type=float, default=0.)
        parser.add_argument('--bce_loss_multiplier', type=float, default=1.)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_learning_rate', type=float, default=1e-7)
        parser.add_argument('--train_batches_to_write', type=int, default=1)
        parser.add_argument('--val_batches_to_write', type=int, default=1)

    def get_model(self):
        model = smp.Unet(encoder_name=self.encoder_name,
                         encoder_weights="imagenet",
                         encoder_depth=self.depth,
                         in_channels=3,
                         classes=self.dataset.num_classes,
                         activation="softmax2d")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(self.dataset.num_channels, 3, (1, 1)),
            torch.nn.BatchNorm2d(3), model)
        return model

    def classwise(self,
                  y_hat,
                  y,
                  weights=None,
                  metric=smp.utils.metrics.IoU(threshold=0.5),
                  dim=0):
        results = torch.empty(y_hat.shape[dim],
                              dtype=y_hat.dtype,
                              device=self.device)
        for i in torch.tensor(range(y_hat.shape[dim]),
                              dtype=torch.long,
                              device=self.device):
            y_hat_class = y_hat.index_select(dim, i)
            y_class = y.index_select(dim, i)
            results[i] = metric(y_hat_class, y_class)

        if weights is not None:
            results = results * weights

        return results.mean()

    def sample_loss(self, y_hat, y, base_multiplier=1.):

        weights = torch.tensor(self.dataset.loss_weights,
                               dtype=y_hat.dtype,
                               device=self.device)

        focal_loss = self.classwise(y_hat,
                                    y,
                                    metric=FocalLoss("binary",
                                                     from_logits=False),
                                    weights=weights,
                                    dim=1)

        dice_loss = self.classwise(y_hat,
                                   y,
                                   metric=smp.losses.DiceLoss(
                                       "binary", from_logits=False),
                                   weights=weights,
                                   dim=1)

        bce_loss = self.classwise(y_hat,
                                  y,
                                  metric=SoftBCELoss(),
                                  weights=weights,
                                  dim=1)

        weighted_bce_loss = base_multiplier * self.bce_loss_multiplier * bce_loss
        weighted_focal_loss = base_multiplier * self.focal_loss_multiplier * focal_loss
        weighted_dice_loss = base_multiplier * self.dice_loss_multiplier * dice_loss

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

    def loss(self, y_hat, y, validation=False):
        batch_loss = 0.

        if self.batch_loss_multiplier > 0:
            assert y_hat.shape[0] % 4 == 0

            num_images = y_hat.shape[0] // 4

            for batch, idx in enumerate(range(0, num_images, 4)):
                y_hat_batch = y_hat[idx:idx + 4, :, :, :]
                y_batch = y[idx:idx + 4, :, :, :]

                y_hat_batch, y_batch = harmonize_batch(y_hat_batch, y_batch)

                weighted_batch_loss_values, unweighted_batch_loss_values = self.sample_loss(
                    y_hat_batch,
                    y_batch,
                    base_multiplier=self.batch_loss_multiplier)

                for k, v in unweighted_batch_loss_values.items():
                    self.log("unweighted_batch_{}_{}".format(batch, k), v)

                for k, v in weighted_batch_loss_values.items():
                    self.log("batch_{}_{}".format(batch, k), v)

                batch_loss += torch.stack(
                    list(weighted_batch_loss_values.values())).sum()

            self.log("batch_loss", batch_loss, prog_bar=True)

        loss = 0.
        if not validation or self.batch_loss_multiplier == 0:
            weighted_loss_values, unweighted_loss_values = self.sample_loss(
                y_hat.clone(), y.clone())

            for k, v in unweighted_loss_values.items():
                self.log("unweighted_{}".format(k), v)

            for k, v in weighted_loss_values.items():
                self.log(k, v, prog_bar=True)

            loss += torch.stack(list(weighted_loss_values.values())).sum()

        return loss + batch_loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=4,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=4,
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
        self.write_predictions(x,
                               y,
                               y_hat,
                               batch_idx,
                               prefix="val",
                               batches_to_write=self.val_batches_to_write)

        if self.batch_loss_multiplier > 0:
            assert y_hat.shape[0] % 4 == 0
            num_images = y_hat.shape[0] // 4

            loss = 0
            batch_ious = []
            for batch, idx in enumerate(range(0, num_images, 4)):
                y_hat_batch = y_hat[idx:idx + 4, :, :, :]
                y_batch = y[idx:idx + 4, :, :, :]
                y_hat_batch, y_batch = harmonize_batch(y_hat_batch, y_batch)

                weighted_batch_loss_values, unweighted_batch_loss_values = self.sample_loss(
                    y_hat_batch, y_batch)

                loss += torch.stack(list(
                    weighted_batch_loss_values.values())).sum()

                y_hat_batch_softmax = torch.nn.Softmax(dim=0)(y_hat_batch)

                iou = self.classwise(y_hat_batch_softmax, y_batch)
                batch_ious.append(iou)
        else:
            loss = self.loss(y_hat, y)
            batch_ious = []
            for batch_idx in range(y_hat.shape[0]):
                iou = self.classwise(y_hat[batch_idx, :, :, :],
                                     y[batch_idx, :, :, :])
                batch_ious.append(iou)

        mean_ious = torch.stack(batch_ious)
        iou = mean_ious.mean()

        self.log("val_iou", iou, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss", loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        image_names = self.test_data.dataset.get_image_names()
        outdir = os.path.join(self.logger.log_dir, "predictions")

        os.makedirs(outdir, exist_ok=True)
        if self.batch_loss_multiplier > 0:
            assert y_hat.shape[0] % 4 == 0
            num_images = y_hat.shape[0] // 4

            for batch, idx in enumerate(range(0, num_images, 4)):
                image_name = image_names[batch_idx * num_images + batch]
                y_hat_batch = y_hat[idx:idx + 4, :, :, :]
                y_batch = y[idx:idx + 4, :, :, :]
                y_hat_batch, y_batch = harmonize_batch(y_hat_batch, y_batch)

                x_batch = x[idx, :, :, :]
                x_batch = x_batch.clone().detach().cpu().numpy().transpose(
                    1, 2, 0)
                y_batch = y_batch.clone().detach().cpu().numpy().transpose(
                    1, 2, 0)
                y_hat_batch = y_hat_batch.clone().detach().cpu().numpy(
                ).transpose(1, 2, 0)

                y_batch = np.argmax(y_batch, axis=2)
                y_hat_batch = np.argmax(y_hat_batch, axis=2)

                outfile = os.path.join(outdir, image_name)
                np.savez_compressed(outfile,
                                    image=x_batch,
                                    mask=y_batch,
                                    predicted_mask=y_hat_batch)

        else:
            batch_size = y_hat.shape[0]

            for i in range(batch_size):
                x_batch = x[
                    i, :, :, :].clone().detach().cpu().numpy().transpose(
                        1, 2, 0)
                y_batch = y[
                    i, :, :, :].clone().detach().cpu().numpy().transpose(
                        1, 2, 0)
                y_hat_batch = y_hat[
                    i, :, :, :].clone().detach().cpu().numpy().transpose(
                        1, 2, 0)

                y_batch = np.argmax(y_batch, axis=2)
                y_hat_batch = np.argmax(y_hat_batch, axis=2)

                image_name = image_names[batch_idx * batch_size + i]
                outfile = os.path.join(outdir, image_name)
                np.savez_compressed(outfile,
                                    image=x_batch,
                                    mask=y_batch,
                                    predicted_mask=y_hat_batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            min_lr=self.min_learning_rate,
            verbose=True,
            mode='max')

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_iou'
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

        if batch_idx >= batches_to_write:
            return

        try:
            self.logger.log_dir
        except AttributeError:
            return

        x = x.clone().detach().cpu()
        y = y.clone().detach().cpu()
        y_hat = y_hat.clone().detach().cpu()

        if self.batch_loss_multiplier > 0:
            assert y_hat.shape[0] % 4 == 0

            num_images = y_hat.shape[0] // 4

            for batch, idx in enumerate(range(0, num_images, 4)):
                x_batch = x[idx:idx + 4, :, :, :]
                y_hat_batch = y_hat[idx:idx + 4, :, :, :]
                y_batch = y[idx:idx + 4, :, :, :]

                x_batch = x_batch[0, :, :, :]
                predicted_mask, mask = harmonize_batch(y_hat_batch, y_batch)

                img = x_batch.numpy().transpose(1, 2, 0)
                predicted_mask = predicted_mask.numpy().transpose(1, 2, 0)
                mask = mask.numpy().transpose(1, 2, 0)

                mask_img = np.argmax(mask, axis=2) * self.intensity
                predicted_mask_img = np.argmax(predicted_mask,
                                               axis=2) * self.intensity

                row_start, row_end, col_start, col_end = crop_image_only_outside(
                    img)

                img = img[row_start:row_end, col_start:col_end, :]
                mask_img = mask_img[row_start:row_end, col_start:col_end]
                predicted_mask_img = predicted_mask_img[row_start:row_end,
                                                        col_start:col_end]

                outfile = os.path.join(self.logger.log_dir,
                                       "{}_{}.png".format(prefix, batch_idx))

                self.save_prediction(img, mask_img, predicted_mask_img,
                                     outfile)

        for i in range(y_hat.shape[0]):
            img = x[i, :, :, :]
            img = img.numpy().transpose(1, 2, 0)

            mask = y[i, :, :, :]
            mask = mask.numpy().transpose(1, 2, 0)
            mask_img = np.argmax(mask, axis=2) * self.intensity

            predicted_mask = y_hat[i, :, :, :]
            predicted_mask = predicted_mask.numpy().transpose(1, 2, 0)
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
