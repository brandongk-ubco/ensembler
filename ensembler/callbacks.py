import pytorch_lightning as pl
import json
import os


class RecordTrainStatus(pl.callbacks.Callback):
    def _write_train_status(self, trainer, pl_module):
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
        with open(os.path.join(trainer.logger.save_dir, "trainer.json"),
                  "w") as statefile:
            json.dump(state, statefile, indent=4)

    def on_train_end(self, trainer, pl_module):
        self._write_train_status(trainer, pl_module)

    def on_epoch_end(self, trainer, pl_module):
        self._write_train_status(trainer, pl_module)
