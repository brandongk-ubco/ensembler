from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch
from ensembler.losses import FocalLoss, TverskyLoss
# from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore, Accuracy
import monai
from ensembler.Activations import Activations
from ensembler.utils import classwise


class Segmenter(LightningModule):
    def __init__(self,
                 in_channels: int = 3,
                 out_classes: int = 1,
                 patience: int = 3,
                 depth: int = 5,
                 width: int = 40,
                 residual_units: int = 2,
                 width_ratio: float = 1.0,
                 focal_loss_multiplier: float = 1.0,
                 tversky_loss_multiplier: float = 1.0,
                 weight_decay: float = 0.0,
                 learning_rate: float = 5e-3,
                 min_learning_rate: float = 5e-4,
                 activation: Activations = "relu"):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model(), torch.nn.Sigmoid())

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=2 * self.hparams.patience,
                          monitor='val_loss',
                          verbose=True,
                          mode='min'),
            ModelCheckpoint(monitor='val_loss',
                            save_top_k=1,
                            mode="min",
                            filename='{epoch}-{val_loss:.6f}'),
        ]

        try:
            callbacks.append(GPUStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def get_model(self):

        channels = [self.hparams.width] * self.hparams.depth
        channels = [
            int(c * self.hparams.width_ratio**i)
            for i, c in enumerate(channels)
        ]

        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_classes,
            channels=channels,
            strides=[2] * self.hparams.depth,
            num_res_units=self.hparams.residual_units,
            # dropout=(monai.networks.layers.factories.Dropout.ALPHADROPOUT, {
            #     "p": self.dropout
            # }),
            act=Activations.get(self.hparams.activation),
            norm=monai.networks.layers.factories.Norm.BATCH)

        return model

    def loss(self, y_hat, y):
        assert y_hat.shape == y.shape

        focal_loss = classwise(
            y_hat,
            y,
            metric=lambda y_hat, y: -1
            if not y.max() > 0 else FocalLoss("binary", from_logits=False)
            (y_hat, y),
            dim=1)

        focal_loss = self.hparams.focal_loss_multiplier * focal_loss[
            focal_loss >= 0].mean()

        tversky_loss = classwise(
            y_hat,
            y,
            metric=lambda y_hat, y: -1
            if not y.max() > 0 else TverskyLoss(from_logits=False)(y_hat, y),
            dim=1)

        tversky_loss = self.hparams.tversky_loss_multiplier * tversky_loss[
            tversky_loss >= 0].mean()

        loss = focal_loss + tversky_loss
        return loss

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_learning_rate,
            verbose=True,
            mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
