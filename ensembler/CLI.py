from pytorch_lightning.utilities.cli import LightningCLI
from ensembler.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
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

        if sys.platform in ['linux', 'linux2']:
            self.config_init['trainer']['plugins'] = DDPPlugin(
                find_unused_parameters=False, sync_batchnorm=True)

        self.trainer = self.trainer_class(**self.config_init['trainer'])
