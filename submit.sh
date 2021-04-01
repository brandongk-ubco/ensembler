#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=6
#SBATCH --time=0-3:00:00
#SBATCH --signal=SIGUSR1@90

module load python/3.7
source ~/envs/ensembler/bin/activate

# debugging flags (optional)
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

srun python -m ensembler voc train