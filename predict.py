from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os
import glob
import re
import time


class TestingCLI(LightningCLI):

    def before_fit(self) -> None:
        """Implement to run some code before fit is started"""

    def fit(self) -> None:
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""

    def after_fit(self) -> None:
        """Implement to run some code after fit has finished"""

    def before_test(self) -> None:
        """Implement to run some code before fit is started"""

    def test(self) -> None:
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        self.trainer.test(**self.fit_kwargs)

    def after_test(self) -> None:
        """Implement to run some code after fit has finished"""


if __name__ == "__main__":

    cli = TestingCLI(
        Segmenter,
        Dataset,
        seed_everything_default=42,
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

    config_dir = os.path.dirname(cli.config["config"][0].abs_path)

    models = glob.glob(
        os.path.join(config_dir, "lightning_logs", "**", "checkpoints",
                     "*.ckpt"))

    model_scores = [
        float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', os.path.basename(m))[-1])
        for m in models
    ]

    model_idx = model_scores.index(min(model_scores))

    model = models[model_idx]

    print("Loading model: {}".format(model))
    version = int(re.findall(r'version_[-+]?([0-9]*\.?[0-9]+)', model)[0])

    logger = TensorBoardLogger(save_dir=config_dir,
                               version=version,
                               name='lightning_logs')

    cli.trainer.logger = logger

    cli.model = cli.model.load_from_checkpoint(model)

    cli.prepare_fit_kwargs()
    cli.before_test()
    start_time = time.clock()
    cli.test()
    end_time = time.clock()
    total_time = end_time - start_time
    with open(os.path.join(config_dir, "predict_time.txt"), "w") as f:
        f.write(str(total_time))
    cli.after_test()
