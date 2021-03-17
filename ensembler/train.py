import environment
import pytorch_lightning as pl
import matplotlib
import sys
from Model import Segmenter as model
import albumentations as A
import UNetEncoder
import os
import json
from torchinfo import summary
import cv2
from parameters import args
from datasets import Datasets

matplotlib.use('Agg')

patience = 10

# class ReRandomCrop(A.RandomCrop):
#     def apply(self, img, h_start=0, w_start=0, **params):
#         print(h_start, w_start)
#         return super().apply(img, h_start, w_start, **params)


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
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    test_transform = A.Compose([
        A.PadIfNeeded(min_height=None,
                      min_width=None,
                      pad_height_divisor=512,
                      pad_width_divisor=512,
                      always_apply=True)
    ])

    return (train_transform, val_transform, test_transform)


if __name__ == '__main__':

    pl.seed_everything(42)

    callbacks = [
        pl.callbacks.EarlyStopping('val_loss', patience=2 * patience),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        RecordTrainStatus()
    ]

    try:
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    except pl.utilities.exceptions.MisconfigurationException:
        pass

    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=1,
                                            callbacks=callbacks,
                                            min_epochs=patience,
                                            deterministic=True,
                                            max_epochs=sys.maxsize)

    dict_args = vars(args)

    trainer.fit(model(get_augments, **dict_args))
