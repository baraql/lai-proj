#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/$MY_USER/lai-proj/logs/loss_ablation_no_fsdp/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 # should match the --nodes parameter
#SBATCH --gpus-per-node=1 # should be up to 4, based on our hardware
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000 # set to maximum to load the biggest models into CPU 
#SBATCH --environment=/iopsstor/scratch/cscs/$MY_USER/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$MY_USER/lai-proj"

# scaling factor = 5 works
# 9 works
# 10 fails
# 11 fails
TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
    --sequence-length 4096 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 100 \
    --scaling-factor 10 \
    --scaling-strategy all \
    --set-seed 42 \
    "
srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$TRAINING_CMD"

echo "END TIME: $(date)"