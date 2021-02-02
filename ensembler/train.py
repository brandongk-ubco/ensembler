import pytorch_lightning as pl
import matplotlib
import sys
from datasets import voc as dataset
from Model import Segmenter as model
import albumentations as A

pl.seed_everything(42)
matplotlib.use('Agg')

patience = 10


def get_augments(image_height, image_width):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(
            image_height,
            image_width,
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True,
                      border_mode=0),
        A.CropNonEmptyMaskIfExists(image_height,
                                   image_width,
                                   always_apply=True)
    ])

    test_transform = A.Compose([])

    return (train_transform, val_transform, test_transform)


if __name__ == '__main__':

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss', patience=2 * patience),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer(gpus=1,
                         callbacks=callbacks,
                         min_epochs=patience,
                         deterministic=True,
                         max_epochs=sys.maxsize,
                         auto_scale_batch_size='binsearch')

    trainer.fit(model(dataset, get_augments, patience=10))
