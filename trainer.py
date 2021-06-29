from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from ensembler.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
import sys


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--project_name', type=str, default=None)
        parser.add_argument('--entity', type=str, default=None)
        parser.add_argument('--name', type=str, default=None)

    def instantiate_trainer(self, *args, **kwargs):
        super().instantiate_trainer(*args, **kwargs)

        self.config_init['trainer']["logger"] = WandbLogger(
            project=self.config["project_name"] if self.config["project_name"]
            else self.config["data"]["dataset"].value,
            entity=self.config["entity"],
            name=self.config["name"],
            resume=True)
        self.trainer = self.trainer_class(**self.config_init['trainer'])


if __name__ == "__main__":

    cli = MyLightningCLI(Segmenter,
                         Dataset,
                         seed_everything_default=42,
                         trainer_defaults={
                             "gpus": -1,
                             "deterministic": True,
                             "max_epochs": sys.maxsize
                         })

    # if "SIGUSR1" in [f.name for f in (signal.Signals)]:
    #     signal.signal(signal.SIGUSR1, wandb.mark_preempting)
