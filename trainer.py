from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from ensembler.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--project_name', type=str, default=None)
        parser.add_argument('--entity', type=str, default=None)
        parser.add_argument('--name', type=str, default=None)


if __name__ == "__main__":

    cli = MyLightningCLI(Segmenter, Dataset)

    import pdb
    pdb.set_trace()

    logger = WandbLogger(
        project=dict_args["project_name"]
        if dict_args["project_name"] else dict_args["dataset_name"],
        entity=dict_args["entity"],
        name=dict_args["name"],
        resume=True)

    # if "SIGUSR1" in [f.name for f in (signal.Signals)]:
    #     signal.signal(signal.SIGUSR1, wandb.mark_preempting)

    # trainer = pl.Trainer.from_argparse_args(args,
    #                                         gpus=-1,
    #                                         callbacks=callbacks,
    #                                         deterministic=True,
    #                                         accelerator="ddp",
    #                                         logger=wandb_logger,
    #                                         move_metrics_to_cpu=True)
