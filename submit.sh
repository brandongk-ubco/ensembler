#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2
#SBATCH --mem=64000M
#SBATCH --cpus-per-task=12
#SBATCH --time=0-3:00:00
#SBATCH --signal=SIGUSR1@90

module load python/3.7
source ~/envs/ensembler/bin/activate

echo "Using temporary storage ${SLURM_TMPDIR}/cityscapes"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 rsync --recursive --progress ${DATA_DIR}/cityscapes ${SLURM_TMPDIR}
echo "Done copying files"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export NUM_WORKERS=11
export OVERRIDE_DATA_DIR=${SLURM_TMPDIR}

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 wandb agent --count 1 $SWEEP_ID
