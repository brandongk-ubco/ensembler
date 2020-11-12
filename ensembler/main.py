import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from argparse import ArgumentParser
from albumentations.pytorch import ToTensorV2
# from datasets.cityscapes import CityscapesAugmentedDataset as AugmentedDataset, train_data, val_data
from datasets.voc import VOCAugmentedDataset as AugmentedDataset, train_data, val_data
import matplotlib
import albumentations as A

from ensembler.UNetEncoder import UNetEncoder

pl.seed_everything(42)
matplotlib.use('Agg')


def get_optimizer():
    return torch.optim.Adam


def get_loss():
    return smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()


def get_model():
    return smp.Unet('unet',
                    in_channels=3,
                    classes=1,
                    activation='sigmoid',
                    encoder_weights=None)


class Segmenter(pl.LightningModule):
    def __init__(self, model, optimizer, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=100,
                                                               verbose=True)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": 'val_loss'
        }


if __name__ == '__main__':

    train_transform = A.Compose([
        A.PadIfNeeded(min_height=128,
                      min_width=128,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(
            128,
            128,
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=128,
                      min_width=128,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(128, 128),
        ToTensorV2()
    ])

    train_data = AugmentedDataset(train_data, train_transform)
    val_data = AugmentedDataset(val_data, val_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=30,
                                               num_workers=8, 
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=30,
                                             num_workers=8,
                                             shuffle=False)

    callbacks = [pl.callbacks.EarlyStopping('val_loss', patience=200), pl.callbacks.LearningRateMonitor(logging_interval='epoch')]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    # logger = pl.loggers.csv_logs.CSVLogger("logs", name="unet")

    trainer = pl.Trainer(gpus=args.gpus,
                         callbacks=callbacks,
                        #  logger=logger,
                         min_epochs=10,
                         deterministic=True,
                         check_val_every_n_epoch=10)

    model = Segmenter(get_model(), get_optimizer(), get_loss())
    trainer.fit(model, train_loader, val_loader)
