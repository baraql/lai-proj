#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:29:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/lai-proj/logs/load_model_no_fsdp/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=ngc_pt_jan     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

set -eo pipefail

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/lai-proj"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $ASSIGNMENT_DIR/load_model_no_fsdp.py"

echo "Running $ASSIGNMENT_DIR/load_model_no_fsdp.py"

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

# srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$PROFILING_CMD"

echo "END TIME: $(date)"