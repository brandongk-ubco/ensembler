from pytorch_lightning.utilities.cli import LightningCLI
from ensembler.loggers import WandbLogger
import sys


class CLI(LightningCLI):
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
        if sys.platform == 'win32':
            self.config_init['trainer']["accelerator"] = None
        else:
            self.config_init['trainer']["accelerator"] = "ddp"
        self.config_init['trainer']["sync_batchnorm"] = True
        self.trainer = self.trainer_class(**self.config_init['trainer'])