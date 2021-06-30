from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from ensembler.CLI import CLI
import sys

if __name__ == "__main__":

    cli = CLI(Segmenter,
              Dataset,
              seed_everything_default=42,
              trainer_defaults={
                  "gpus": -1,
                  "deterministic": True,
                  "max_epochs": sys.maxsize,
                  "sync_batchnorm": True
              })

    # if "SIGUSR1" in [f.name for f in (signal.Signals)]:
    #     signal.signal(signal.SIGUSR1, wandb.mark_preempting)
