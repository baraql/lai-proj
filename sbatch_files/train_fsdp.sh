#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/%u/lai-proj/logs/train_fsdp/%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=2 # should match the --nodes parameter
#SBATCH --gpus-per-node=4 # should be up to 4, based on our hardware
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000 # set to maximum to load the biggest models into CPU 
#SBATCH --environment=ngc_pt_jan     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# MEMORY INFO
# GPU 0: NVIDIA GH200 120GB
#   Total memory: 94.50 GB
# RAM:
#   shared per node: 776.10 GB
#   depending on the number of processes per node: 194 GB - 776.10 GB

set -eo pipefail

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/lai-proj"


# https://gitlab.uzh.ch/s3it/docs/-/blob/issue80/docs/cluster/python_gpu_example.md?ref_type=heads

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip

# If the code fails with:
#  [W521 23:55:23.120749489 TCPStore.cpp:115] [c10d] recvVector failed on SocketImpl(fd=3, addr=[nid007108]:44074, remote=[nid007089]:29500): failed to recv, got 0 bytes
# Exception raised from recvBytes at /opt/pytorch/pytorch/torch/csrc/distributed/c10d/Utils.hpp:671 (most recent call first):
# 
# Then (i believe) the port is already used so need to change to another port (eg 29505, 29500)
srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29505 \
    $ASSIGNMENT_DIR/train_fsdp.py \
      --sequence-length 4096 \
      --batch-size 1 \
      --learning-rate 5e-5 \
      --lr-warmup-steps 100 \
      --training-steps 100 \
      --scaling-factor 6 \
      --scaling-strategy all \
      --set-seed 42

echo "END TIME: $(date)"
