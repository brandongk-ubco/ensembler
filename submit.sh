#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --signal=SIGUSR1@90

python -m ensembler "$DATASET_NAME" train --num_workers 5