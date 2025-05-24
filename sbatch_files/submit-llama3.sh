#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/lai-proj/logs/flash_attention/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=my_env     # Vanilla 25.01 PyTorch NGC Image
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/lai-proj"

CMD_PREFIX="numactl --membind=0-3"

CUDNN_LOGERR_DBG=1
CUDNN_LOGDEST_DBG=stderr

TRAINING_CMD="python3 $ASSIGNMENT_DIR/train.py \
    --sequence-length 2048 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --fused-optimizer \
    --fused-attention \
    "

# PROFILING_CMD="nsys profile -s none -w true \
# --trace='nvtx,cudnn,cublas,cuda' \
# --output=/iopsstor/scratch/cscs/$USER/assignment-2/nsys-trace.nsys-rep \
# --force-overwrite true \
# --capture-range=cudaProfilerApi \
# --capture-range-end=stop -x true numactl --membind=0-3 python3 $ASSIGNMENT_DIR/train.py --profile"

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "pip install transformers && $CMD_PREFIX $TRAINING_CMD"

#srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$PROFILING_CMD"

echo "END TIME: $(date)"