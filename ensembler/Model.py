import pytorch_lightning as pl
import torch
from ensembler.losses import FocalLoss, TverskyLoss
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore, Accuracy
import monai
from ensembler.Activations import Activations


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
        self.activation = Activations.get(self.hparams["activation"])

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
        parser.add_argument('--width_ratio', type=float, default=1.2)
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
        parser.add_argument('--activation',
                            type=str,
                            choices=Activations.choices(),
                            default="relu")

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
            act=self.activation,
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

    def sample_loss(self, y_hat, y):

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
                "focal_loss"] = self.focal_loss_multiplier * focal_loss[
                    focal_loss >= 0].mean()

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
                "tversky_loss"] = self.tversky_loss_multiplier * tversky_loss[
                    tversky_loss >= 0].mean()

        return loss_values["focal_loss"] + loss_values["tversky_loss"]

    def loss(self, y_hat, y):

        return self.sample_loss(y_hat, y)

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

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def training_step_end(self, steps):
        # Hack to handle the single step case
        if type(steps) == torch.Tensor:
            steps = [steps]

        return torch.stack(steps).mean()

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def validation_epoch_end(self, steps):

        if type(steps) == torch.Tensor:
            steps = [steps]

        self.log("val_loss", torch.stack(steps).mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        # steps_per_epoch = self.trainer.limit_train_batches // self.trainer.accumulate_grad_batches
        # epochs = self.trainer.max_epochs
        # total_steps = steps_per_epoch * epochs
        # warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        # cooldown_steps = int(self.cooldown_epochs * steps_per_epoch)
        # lr_controlled_steps = total_steps - cooldown_steps

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=warmup_steps,
        #     max_epochs=lr_controlled_steps,
        #     eta_min=self.min_learning_rate)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, eta_min=self.min_learning_rate, T_max=total_steps)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            cooldown=self.patience // 2,
            min_lr=self.min_learning_rate,
            mode="min")

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
                "monitor": "val_loss",
                # "interval": "step"
            }
        }
