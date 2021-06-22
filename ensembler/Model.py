import pytorch_lightning as pl
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from ensembler.losses import FocalLoss, TverskyLoss
from ensembler.utils import crop_image_only_outside
from ensembler.aggregators import harmonize_batch
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore, Accuracy
import monai
from ensembler.optim import PolynomialLRDecayWithWarmup, LinearWarmupCosineAnnealingLR


class Segmenter(pl.LightningModule):
    def __init__(self, dataset, train_data, val_data, test_data, batch_size,
                 **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.patience = self.hparams["patience"]
        self.encoder_name = self.hparams["encoder_name"]
        self.num_workers = self.hparams["num_workers"]
        self.depth = self.hparams["depth"]
        self.width = self.hparams["width"]
        self.width_ratio = self.hparams["width_ratio"]
        self.batch_size = batch_size
        self.batch_loss_multiplier = self.hparams["batch_loss_multiplier"]
        self.base_loss_multiplier = self.hparams.get("base_loss_multiplier",
                                                     1.)
        self.focal_loss_multiplier = self.hparams["focal_loss_multiplier"]
        self.weight_decay = self.hparams["weight_decay"]
        self.learning_rate = self.hparams["learning_rate"]
        self.min_learning_rate = self.hparams["min_learning_rate"]
        self.warmup_epochs = self.hparams["warmup_epochs"]
        self.cooldown_epochs = self.hparams["cooldown_epochs"]
        self.train_batches_to_write = self.hparams["train_batches_to_write"]
        self.val_batches_to_write = self.hparams["val_batches_to_write"]
        self.tversky_loss_multiplier = self.hparams["tversky_loss_multiplier"]
        self.patience = self.hparams["patience"]
        self.residual_units = self.hparams["residual_units"]
        self.learning_rate_decay_power = self.hparams[
            "learning_rate_decay_power"]

        self.dropout = self.hparams["dropout"]

        self.dataset = dataset
        self.val_data = val_data
        self.train_data = train_data
        self.test_data = test_data

        self.intensity = 255 // (self.dataset.num_classes + 1)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(self.dataset.num_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model(), torch.nn.Sigmoid())

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--encoder_name',
                            type=str,
                            default="efficientnet-b0")
        parser.add_argument('--depth', type=int, default=10)
        parser.add_argument('--width', type=int, default=40)
        parser.add_argument('--width_ratio', type=int, default=1.2)
        parser.add_argument('--dropout', type=float, default=0)

        parser.add_argument('--focal_loss_multiplier', type=float, default=1.)

        parser.add_argument('--tversky_loss_multiplier',
                            type=float,
                            default=1.)

        parser.add_argument('--residual_units', type=int, default=2)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--learning_rate', type=float, default=5e-3)
        parser.add_argument('--min_learning_rate', type=float, default=3e-4)
        parser.add_argument('--train_batches_to_write', type=int, default=1)
        parser.add_argument('--val_batches_to_write', type=int, default=4)
        parser.add_argument('--warmup_epochs', type=float, default=0.05)
        parser.add_argument('--cooldown_epochs', type=float, default=0.)
        parser.add_argument('--learning_rate_decay_power',
                            type=float,
                            default=2.)

    def get_model(self):

        channels = [self.width] * self.depth
        channels = [
            int(c * self.width_ratio**i) for i, c in enumerate(channels)
        ]

        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=3,
            out_channels=self.dataset.num_classes,
            channels=channels,
            strides=[2] * self.depth,
            num_res_units=self.residual_units,
            dropout=(monai.networks.layers.factories.Dropout.ALPHADROPOUT, {
                "p": self.dropout
            }),
            act=monai.networks.layers.factories.Act.MEMSWISH,
            norm=monai.networks.layers.factories.Norm.BATCH)

        return model

    def classwise(self, y_hat, y, metric, weights=None, dim=1):

        results = torch.empty(y_hat.shape[dim],
                              dtype=y_hat.dtype,
                              device=y_hat.device)

        for i in torch.tensor(range(y_hat.shape[dim]),
                              dtype=torch.long,
                              device=y_hat.device):
            y_hat_class = y_hat.index_select(dim, i)
            y_class = y.index_select(dim, i)
            results[i] = metric(y_hat_class, y_class)

        if weights is not None:
            results = results * weights

        return results

    def sample_loss(self, y_hat, y, base_multiplier=1.):

        weights = torch.tensor(self.dataset.loss_weights,
                               dtype=y_hat.dtype,
                               device=y_hat.device)

        loss_values = {}

        if self.focal_loss_multiplier > 0:
            focal_loss = self.classwise(
                y_hat,
                y,
                metric=lambda y_hat, y: -1
                if not y.max() > 0 else FocalLoss("binary", from_logits=False)
                (y_hat, y),
                weights=weights,
                dim=1)

            loss_values[
                "focal_loss"] = base_multiplier * self.focal_loss_multiplier * focal_loss[
                    focal_loss >= 0]

        if self.tversky_loss_multiplier > 0:
            tversky_loss = self.classwise(
                y_hat,
                y,
                metric=lambda y_hat, y: -1
                if not y.max() > 0 else TverskyLoss(from_logits=False)
                (y_hat, y),
                weights=weights,
                dim=1)

            loss_values[
                "tversky_loss"] = base_multiplier * self.tversky_loss_multiplier * tversky_loss[
                    tversky_loss >= 0]

        if self.focal_loss_multiplier > 0 and self.tversky_loss_multiplier > 0:
            loss_values["combined_loss"] = loss_values["focal_loss"].mean(
            ) + loss_values["tversky_loss"].mean()
        elif self.focal_loss_multiplier > 0:
            loss_values["combined_loss"] = loss_values["focal_loss"].mean()
        elif self.tversky_loss_multiplier > 0:
            loss_values["combined_loss"] = loss_values["tversky_loss"].mean()
        else:
            loss_values["combined_loss"] = 0
        return loss_values

    def loss(self, y_hat, y, validation=False):

        results = {}
        loss = torch.tensor(0., dtype=y.dtype, device=y.device)

        if self.base_loss_multiplier > 0. and (
                not validation or self.batch_loss_multiplier == 0):
            loss_values = self.sample_loss(
                y_hat.clone(),
                y.clone(),
                base_multiplier=self.base_loss_multiplier)

            for k, v in loss_values.items():
                results[k] = v

            loss = results["combined_loss"]

        if self.batch_loss_multiplier > 0:
            y_hat_batch, y_batch = self.combine_batch(y_hat, y)

            loss_values = self.sample_loss(y_hat_batch,
                                           y_batch,
                                           base_multiplier=1. if validation
                                           else self.batch_loss_multiplier)

            for k, v in loss_values.items():
                results["batch_{}".format(k)] = v

            if validation:
                loss = results["batch_combined_loss"]
            else:
                loss += results["batch_combined_loss"]

        results["loss"] = loss
        return results

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

        results = {}

        results.update(
            dict(
                zip([
                    "{}_observation_ratio".format(n)
                    for n in self.dataset.classes
                ],
                    y.sum(dim=(0, 2, 3)) / torch.numel(y) *
                    self.dataset.num_classes)))

        results.update(
            dict(
                zip([
                    "{}_observation_frequency".format(n)
                    for n in self.dataset.classes
                ], y.amax(dim=(0, 2, 3)))))

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        results.update(loss)
        return results

    def training_step_end(self, steps):

        # Hack to handle the single step case
        if type(steps) == dict:
            steps = [steps]

        losses = []

        results = {}
        for step in steps:
            losses.append(step["loss"])
            for metric, metric_results in step.items():
                if metric == "loss":
                    continue
                if metric not in results:
                    results[metric] = []
                results[metric].append(metric_results)
        for metric, metric_results in results.items():
            self.log(metric,
                     torch.stack(metric_results).mean(),
                     on_step=True,
                     on_epoch=False)

        return torch.stack(losses).mean()

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
                                  device=y_hat.device)
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

        if self.batch_loss_multiplier > 0:
            y_hat, y = self.combine_batch(y_hat, y)

        metrics = {
            "iou":
            lambda y_hat, y: -1
            if not y.max() > 0 else IoU(threshold=0.5)(y_hat, y),
            "f1":
            lambda y_hat, y: -1
            if not y.max() > 0 else Fscore(threshold=0.5)(y_hat, y),
            "accuracy":
            lambda y_hat, y: -1
            if not y.max() > 0 else Accuracy(threshold=0.5)(y_hat, y),
            "recall":
            lambda y_hat, y: -1
            if not y.max() > 0 else Recall(threshold=0.5)(y_hat, y),
            "precision":
            lambda y_hat, y: -1
            if not y.max() > 0 else Precision(threshold=0.5)(y_hat, y),
            "prediction_ratio":
            lambda y_hat, y: (y_hat > 0.5).sum() / torch.numel(y_hat)
        }

        results = {}

        for metric, metric_func in metrics.items():
            result = self.classwise(y_hat, y, dim=1, metric=metric_func)

            results_list = list(
                zip(["{}_{}".format(n, metric) for n in self.dataset.classes],
                    result))
            results_list = [(r[0], r[1]) for r in results_list if r[1] >= 0]

            results.update(dict(results_list))

        results.update(loss)

        return results

    def validation_epoch_end(self, steps):

        if type(steps) == dict:
            steps = [steps]

        losses = []

        results = {}

        for step in steps:
            losses.append(step["loss"])
            for metric, metric_result in step.items():
                if metric == "loss":
                    continue
                if metric not in results:
                    results[metric] = []
                results[metric].append(metric_result)
        for metric, metric_results in results.items():
            self.log("val_{}".format(metric),
                     torch.stack(metric_results).mean())
        self.log("val_loss", torch.stack(losses).mean())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.sigmoid(y_hat)

        assert y_hat.shape[0] % 4 == 0

        y[1, :, :, :] = torch.flip(y[1, :, :, :], [1])
        y[2, :, :, :] = torch.flip(y[2, :, :, :], [2])
        y[3, :, :, :] = torch.flip(y[3, :, :, :], [1, 2])

        y_hat[1, :, :, :] = torch.flip(y_hat[1, :, :, :], [1])
        y_hat[2, :, :, :] = torch.flip(y_hat[2, :, :, :], [2])
        y_hat[3, :, :, :] = torch.flip(y_hat[3, :, :, :], [1, 2])

        assert torch.max(y_hat) >= 0
        assert torch.max(y_hat) <= 1
        assert torch.max(y) >= 0
        assert torch.max(y) <= 1

        assert torch.allclose(y[1, :, :, :], y[0, :, :, :])
        assert torch.allclose(y[2, :, :, :], y[0, :, :, :])
        assert torch.allclose(y[3, :, :, :], y[0, :, :, :])

        image_names = self.test_data.dataset.get_image_names()
        out_dir = self.logger.log_dir
        outdir = os.path.join(out_dir, "predictions")
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

        steps_per_epoch = self.trainer.limit_train_batches // self.trainer.accumulate_grad_batches
        epochs = self.trainer.max_epochs
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        cooldown_steps = int(self.cooldown_epochs * steps_per_epoch)
        lr_controlled_steps = total_steps - cooldown_steps

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=lr_controlled_steps,
            eta_min=self.min_learning_rate)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, eta_min=self.min_learning_rate, T_max=total_steps)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     patience=self.patience,
        #     cooldown=self.patience // 2,
        #     min_lr=self.min_learning_rate,
        #     mode="min")

        # scheduler = PolynomialLRDecayWithWarmup(
        #     optimizer,
        #     total_steps=total_steps,
        #     warmup_steps=warmup_steps,
        #     decay_power=self.learning_rate_decay_power,
        #     min_lr=self.min_learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "monitor": "val_loss",
                "interval": "step"
            }
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
