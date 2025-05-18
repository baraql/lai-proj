# Large Scale AI Engineering – FSDP Project

This repository contains the project for the ETH course **Large Scale AI Engineering**. It implements **Fully Sharded Data Parallel (FSDP)** training for a Transformer model and runs experiments on a multi-GPU cluster.


## Project Structure
- `sbatch_files/` — contains Slurm job scripts

- `*.py` — training, evaluation, and model code

-  `logs/` — output logs (written per job)


## 🚀 Getting Started
 Since `#SBATCH` directives do not expand shell variables like `$USER`, to ensure a smooth running you first need to export your username as an environment variable `$MY_USER` by running:
 ```bash
 $ export MY_USER=your_clariden_username
 ```

 You can verify that it's set by running: 
 ```bash
 $ echo MY_USER
 ```

 Then to submit any sbatch script that uses `$MY_USER` in the directives, simply run:
 ```bash
envsubst '$MY_USER' < sbatch_file/your_sbatch_script.sh | sbatch
 ```

 Otherwise it won't work unless we hardcode usernames which we don't want to!

## Creating `sbatch` files
Please, create `sbatch` files only inside the `sbatch_files` folder. 
Specify a separate directory for logs that would contain outputs for this job only, e.g. `fsdp-experiments` inside `logs` folder. 
Don't hardcode any usernames in your `sbatch` files, use `$MY_USER` instead.

Here is an example of an `sbatch` script: 
```bash
#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --job-name=lsai
#SBATCH --output=/iopsstor/scratch/cscs/$MY_USER/lai-proj/logs/fsdp-experiments/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/$MY_USER/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

set -eo pipefail

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$MY_USER/lai-proj"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $ASSIGNMENT_DIR/load_model_no_fsdp.py"

echo "Running $ASSIGNMENT_DIR/load_model_no_fsdp.py"

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

# srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$PROFILING_CMD"

echo "END TIME: $(date)"
```



## Experiment 1: maximum model size that fits on a single GPU without FSDP 

First, we establish the biggest model that can fit into a single GPU wihtout FSDP. For that we run a binary search changing the number of layers in the model until we find the best fit.

**Results**: the biggest model has 48,185,937,920 parameters, achieved with dim=4096, n_layers=216, n_heads=32

**Implementation**: `scratch/lai-proj/load_model_no_fsdp.py` \
**Sbatch file**: `/users/elyulina/scratch/lai-proj/sbatch_files/load_model_no_fsdp.sh` \
**Log file**: `scratch/lai-proj/logs/load_model_no_fsdp/lsai-446975.out`



