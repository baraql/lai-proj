#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:29:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/$MY_USER/lai-proj/logs/load_model_fsdp/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 # should match the --nodes parameter
#SBATCH --gpus-per-node=4 # should be up to 4, based on our hardware
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000 # set to maximum to load the biggest models into CPU 
#SBATCH --environment=/iopsstor/scratch/cscs/$MY_USER/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
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
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$MY_USER/lai-proj"


# https://gitlab.uzh.ch/s3it/docs/-/blob/issue80/docs/cluster/python_gpu_example.md?ref_type=heads

# Node networking section
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip

# If the code fails with:
#  [W521 23:55:23.120749489 TCPStore.cpp:115] [c10d] recvVector failed on SocketImpl(fd=3, addr=[nid007108]:44074, remote=[nid007089]:29500): failed to recv, got 0 bytes
# Exception raised from recvBytes at /opt/pytorch/pytorch/torch/csrc/distributed/c10d/Utils.hpp:671 (most recent call first):
# 
# Then (i believe) the port is already used so need to change to another port (eg 29505)
srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29505 \
    $ASSIGNMENT_DIR/load_model_fsdp.py 


# export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# export MASTER_PORT=29500                     # any free port is fine
# export WORLD_SIZE=$(($SLURM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))
# export RANK=$SLURM_PROCID                    # unique rank per process

# srun python -m torch.distributed.run \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --nproc_per_node=$SLURM_GPUS_PER_NODE \
#     --node_rank=$SLURM_NODEID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     $ASSIGNMENT_DIR/load_model_fsdp.py 

echo "END TIME: $(date)"
