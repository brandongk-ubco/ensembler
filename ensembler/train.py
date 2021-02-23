import pytorch_lightning as pl
import matplotlib
import sys
from datasets import severstal as dataset
from Model import Segmenter as model
import albumentations as A
import UNetEncoder
import os
import json
from torchinfo import summary

pl.seed_everything(42)
matplotlib.use('Agg')

patience = 10


class RecordTrainStatus(pl.callbacks.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        state = {
            "Trainer": {
                "state": trainer.state.value,
                "current_epoch": trainer.current_epoch,
                "num_gpus": trainer.num_gpus,
                "max_epochs": trainer.max_epochs,
                "max_steps": trainer.max_steps,
                "min_epochs": trainer.min_epochs,
                "min_steps": trainer.min_steps,
            },
            "EarlyStopping": {
                "best_score": float(trainer.callbacks[0].best_score),
                "patience": float(trainer.callbacks[0].patience),
                "wait_count": float(trainer.callbacks[0].wait_count),
                "stopped_epoch": float(trainer.callbacks[0].stopped_epoch),
                "min_delta": float(trainer.callbacks[0].min_delta),
            },
            "Scheduler": trainer.lr_schedulers[0]["scheduler"].state_dict()
        }
        with open(os.path.join(trainer.logger.log_dir, "trainer.json"),
                  "w") as statefile:
            json.dump(state, statefile, indent=4)


class ModelSummary(pl.callbacks.Callback):
    def on_sanity_check_end(self, trainer, pl_module):
        dataloader = trainer.val_dataloaders[0]
        batch = next(iter(dataloader))
        model_summary = summary(trainer.model,
                                input_size=tuple(batch[0].shape),
                                verbose=0)
        print(model_summary)
        with open(os.path.join(trainer.logger.log_dir, "model.txt"),
                  "w") as modelfile:
            modelfile.write(str(model_summary))


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
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        RecordTrainStatus()
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer(gpus=1,
                         callbacks=callbacks,
                         min_epochs=patience,
                         deterministic=True,
                         max_epochs=sys.maxsize)

    trainer.fit(model(dataset, get_augments, patience=10))
