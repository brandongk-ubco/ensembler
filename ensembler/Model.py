import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from ensembler.losses import FocalLoss, SoftBCELoss, TverskyLoss
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
        self.focal_loss_gamma = self.hparams["focal_loss_gamma"]
        self.weight_decay = self.hparams["weight_decay"]
        self.learning_rate = self.hparams["learning_rate"]
        self.min_learning_rate = self.hparams["min_learning_rate"]
        self.train_batches_to_write = self.hparams["train_batches_to_write"]
        self.val_batches_to_write = self.hparams["val_batches_to_write"]
        self.final_activation = self.hparams["final_activation"]
        self.tversky_loss_multiplier = self.hparams["tversky_loss_multiplier"]
        self.tversky_loss_alpha = self.hparams["tversky_loss_alpha"]
        self.tversky_loss_beta = self.hparams["tversky_loss_beta"]
        self.tversky_loss_gamma = self.hparams["tversky_loss_gamma"]
        self.dataset = dataset
        self.val_data = val_data
        self.train_data = train_data
        self.test_data = test_data

        self.intensity = 255 // (self.dataset.num_classes + 1)

        self.model = self.get_model()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--encoder_name',
                            type=str,
                            default="efficientnet-b3")
        parser.add_argument('--depth', type=int, default=5)

        parser.add_argument('--focal_loss_multiplier', type=float, default=1.)
        parser.add_argument('--focal_loss_gamma', type=float, default=2.)

        parser.add_argument('--tversky_loss_multiplier',
                            type=float,
                            default=1.)
        parser.add_argument('--tversky_loss_alpha', type=float,
                            default=0.5)  # Penalty for False Positives
        parser.add_argument('--tversky_loss_beta', type=float,
                            default=2)  # Penalty for False Negative
        parser.add_argument('--tversky_loss_gamma', type=float, default=2)

        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_learning_rate', type=float, default=1e-7)
        parser.add_argument('--train_batches_to_write', type=int, default=1)
        parser.add_argument('--val_batches_to_write', type=int, default=1)
        parser.add_argument('--final_activation', type=str, default=None)

    def get_model(self):
        model = smp.Unet(encoder_name=self.encoder_name,
                         encoder_weights="imagenet",
                         encoder_depth=self.depth,
                         in_channels=3,
                         classes=self.dataset.num_classes,
                         activation=self.final_activation)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(self.dataset.num_channels, 3, (1, 1)),
            torch.nn.BatchNorm2d(3), model)
        return model

    def classwise(self,
                  y_hat,
                  y,
                  weights=None,
                  metric=smp.utils.metrics.IoU(threshold=0.5),
                  dim=1):

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

        return results

    def sample_loss(self, y_hat, y, base_multiplier=1.):

        weights = torch.tensor(self.dataset.loss_weights,
                               dtype=y_hat.dtype,
                               device=self.device)

        loss_values = {}

        if self.focal_loss_multiplier > 0:
            loss_values[
                "focal_loss"] = base_multiplier * self.focal_loss_multiplier * self.classwise(
                    y_hat,
                    y,
                    metric=FocalLoss("binary",
                                     from_logits=True,
                                     gamma=self.focal_loss_gamma),
                    weights=weights,
                    dim=1).mean()

        if self.tversky_loss_multiplier > 0:
            loss_values[
                "tversky_loss"] = base_multiplier * self.tversky_loss_multiplier * self.classwise(
                    y_hat,
                    y,
                    metric=TverskyLoss(alpha=self.tversky_loss_alpha,
                                       beta=self.tversky_loss_beta,
                                       gamma=self.tversky_loss_gamma,
                                       from_logits=True),
                    weights=weights,
                    dim=1).mean()

        return loss_values

    def loss(self, y_hat, y, validation=False):

        prefix = "val_" if validation else ""

        if not validation or self.batch_loss_multiplier == 0:
            loss_values = self.sample_loss(y_hat.clone(), y.clone())

            for k, v in loss_values.items():
                self.log("{}{}".format(prefix, k), v)

            loss = torch.stack(list(loss_values.values())).sum()

        if self.batch_loss_multiplier > 0:
            y_hat_batch, y_batch = self.combine_batch(y_hat, y)

            loss_values = self.sample_loss(
                y_hat_batch,
                y_batch,
                base_multiplier=1. + self.batch_loss_multiplier
                if validation else self.batch_loss_multiplier)

            for k, v in loss_values.items():
                self.log("{}batch_loss_{}".format(prefix, k), v)

            batch_loss = torch.stack(list(loss_values.values())).sum()

            self.log("{}batch_loss".format(prefix), batch_loss, prog_bar=True)

            if validation:
                loss = batch_loss
            else:
                loss += batch_loss

        return loss

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

        self.log_dict(dict(
            zip([
                "{}_observation_ratio".format(n) for n in self.dataset.classes
            ],
                y.sum(dim=(0, 2, 3)) / torch.numel(y) *
                self.dataset.num_classes)),
                      on_step=False,
                      on_epoch=True)

        self.log_dict(dict(
            zip([
                "{}_observation_frequency".format(n)
                for n in self.dataset.classes
            ], y.amax(dim=(0, 2, 3)))),
                      on_step=False,
                      on_epoch=True)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.write_predictions(x,
                               y,
                               y_hat,
                               batch_idx,
                               prefix="train",
                               batches_to_write=self.train_batches_to_write)

        return loss

    def forward(self, x):
        return self.model(x)

    def combine_batch(self, y_hat, y):
        assert torch.max(y_hat) >= 0
        assert torch.max(y_hat) <= 1
        assert torch.max(y) >= 0
        assert torch.max(y) <= 1

        assert y_hat.shape[0] % 4 == 0
        num_images = y_hat.shape[0] // 4

        new_shape = list(y_hat.size())
        new_shape[0] = num_images

        y_hat_batch = torch.empty(tuple(new_shape),
                                  dtype=y_hat.dtype,
                                  device=self.device)
        y_batch = torch.empty_like(y_hat_batch)

        for batch, idx in enumerate(range(0, y_hat.shape[0], 4)):
            new_y_hat, new_y = harmonize_batch(y_hat[idx:idx + 4, :, :, :],
                                               y[idx:idx + 4, :, :, :])

            y_hat_batch[batch, :, :, :] = new_y_hat.unsqueeze(0)
            y_batch[batch, :, :, :] = new_y.unsqueeze(0)

        assert torch.max(y_hat_batch) >= 0
        assert torch.max(y_hat_batch) <= 1
        assert torch.max(y_batch) >= 0
        assert torch.max(y_batch) <= 1

        return y_hat_batch, y_batch

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y, validation=True)

        self.log_dict({"val_loss": loss}, prog_bar=True)

        if self.batch_loss_multiplier > 0:
            y_hat, y = self.combine_batch(y_hat, y)

        self.write_predictions(x,
                               y,
                               y_hat,
                               batch_idx,
                               prefix="val",
                               batches_to_write=self.val_batches_to_write)

        metrics = {
            "iou":
            smp.utils.metrics.IoU(threshold=0.5),
            "f1":
            smp.utils.metrics.Fscore(threshold=0.5),
            "accuracy":
            smp.utils.metrics.Accuracy(threshold=0.5),
            "recall":
            smp.utils.metrics.Recall(threshold=0.5),
            "precision":
            smp.utils.metrics.Precision(threshold=0.5),
            "prediction_ratio ":
            lambda y_hat, y: (y_hat > 0.5).sum() / torch.numel(y_hat)
        }

        metric_results = {}

        for metric, metric_func in metrics.items():
            result = self.classwise(y_hat, y, dim=1, metric=metric_func)

            self.log_dict(
                dict(
                    zip([
                        "{}_val_{}".format(n, metric)
                        for n in self.dataset.classes
                    ], result)))
            metric_results[metric] = result.mean()

        self.log_dict(
            dict(
                zip(["val_{}".format(n) for n in metric_results.keys()],
                    metric_results.values())))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        assert y_hat.shape[0] % 4 == 0

        y_hat[1, :, :, :] = torch.flip(y_hat[1, :, :, :], [1])
        y_hat[2, :, :, :] = torch.flip(y_hat[2, :, :, :], [2])
        y_hat[3, :, :, :] = torch.flip(y_hat[3, :, :, :], [1, 2])

        image_names = self.test_data.dataset.get_image_names()
        outdir = os.path.join(self.logger.save_dir, "predictions")
        os.makedirs(outdir, exist_ok=True)

        image_name = image_names[batch_idx]

        for idx in range(y_hat.shape[0]):
            prediction = y_hat[
                idx, :, :, :].clone().detach().cpu().numpy().transpose(
                    1, 2, 0)
            prediction = (prediction * 255).astype(np.uint8)

            outfile = os.path.join(outdir, "{}_{}".format(image_name, idx))
            np.savez_compressed(outfile, predicted_mask=prediction)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            min_lr=self.min_learning_rate,
            verbose=True,
            mode='min')

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

        if batch_idx >= batches_to_write:
            return

        try:
            self.logger.save_dir
        except AttributeError:
            return

        x = x.clone().detach().cpu()
        y = y.clone().detach().cpu()
        y_hat = y_hat.clone().detach().cpu()

        for i in range(y_hat.shape[0]):
            img = x[i, :, :, :]
            img = img.numpy().transpose(1, 2, 0)

            mask = y[i, :, :, :]
            mask = mask.numpy().transpose(1, 2, 0)
            mask_img = (np.argmax(mask, axis=2) + 1) * self.intensity

            predicted_mask = y_hat[i, :, :, :]
            predicted_mask = predicted_mask.numpy().transpose(1, 2, 0)
            predicted_mask_img = (np.argmax(predicted_mask, axis=2) +
                                  1) * self.intensity

            row_start, row_end, col_start, col_end = crop_image_only_outside(
                img)

            img = img[row_start:row_end, col_start:col_end, :]
            mask_img = mask_img[row_start:row_end, col_start:col_end]
            predicted_mask_img = predicted_mask_img[row_start:row_end,
                                                    col_start:col_end]

            outfile = os.path.join(self.logger.save_dir,
                                   "{}_{}_{}.png".format(prefix, batch_idx, i))

            self.save_prediction(img, mask_img, predicted_mask_img, outfile)
