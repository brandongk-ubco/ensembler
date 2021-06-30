from ensembler.Model import Segmenter
from ensembler.Dataset import Dataset
from ensembler.CLI import CLI
import sys
import signal
import wandb

if __name__ == "__main__":

    if "SIGUSR1" in [f.name for f in (signal.Signals)]:
        print("Initializing Signal for wandb to mark preempting")
        signal.signal(signal.SIGUSR1, wandb.mark_preempting)

    cli = CLI(Segmenter,
              Dataset,
              seed_everything_default=42,
              trainer_defaults={
                  "gpus": -1,
                  "deterministic": True,
                  "max_epochs": sys.maxsize,
                  "sync_batchnorm": True
              })
