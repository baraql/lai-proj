#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/$MY_USER/lai-proj/logs/loss_ablation_fsdp/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/$MY_USER/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$MY_USER/lai-proj"

CMD_PREFIX="numactl --membind=0-3"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

srun --cpus-per-task=$SLURM_CPUS_PER_TASK \
  python3 -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    $ASSIGNMENT_DIR/train_fsdp.py \
      --sequence-length 4096 \
      --batch-size 1 \
      --learning-rate 5e-5 \
      --lr-warmup-steps 100 \
      --training-steps 1000 \
      --scale 1 \
      --set-seed 42 \

echo "END TIME: $(date)"
