from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.cli import LightningCLI as CLI
from pytorch_lightning.utilities.cli import SaveConfigCallback
import sys
import os


class MySaveConfigCallback(SaveConfigCallback):

    def on_train_start(self, trainer: Trainer,
                       pl_module: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config,
                         config_path,
                         skip_none=False,
                         overwrite=True)


if __name__ == "__main__":

    cli = CLI(Segmenter,
              Dataset,
              seed_everything_default=42,
              save_config_callback=MySaveConfigCallback,
              trainer_defaults={
                  "gpus":
                      -1,
                  "deterministic":
                      True,
                  "max_epochs":
                      sys.maxsize,
                  "accelerator":
                      "ddp" if sys.platform in ["linux", "linux2"] else None,
                  "sync_batchnorm":
                      True,
              })
