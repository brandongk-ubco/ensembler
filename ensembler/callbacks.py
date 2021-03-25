import pytorch_lightning as pl
import json
import os


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


# class ModelSummary(pl.callbacks.Callback):
#     def on_sanity_check_end(self, trainer, pl_module):
#         dataloader = trainer.val_dataloaders[0]
#         batch = next(iter(dataloader))
#         model_summary = summary(trainer.model,
#                                 input_size=tuple(batch[0].shape),
#                                 verbose=0)
#         print(model_summary)
#         with open(os.path.join(trainer.logger.log_dir, "model.txt"),
#                   "w") as modelfile:
#             modelfile.write(str(model_summary))
