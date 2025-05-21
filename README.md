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
 $ echo $MY_USER
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

First, we establish the biggest model that can fit into a single GPU wihtout FSDP. For that we run a binary search scaling model's parameters until we find the best fit. We also implemented another scaling strategy that only changes the number of layers, allowing for more flexibility and therefore fitting a bigger model at the expense of its architecture. 

**Results**:
Scaling all parameters, the biggest model has **46,322,328,320** parameters, achieved with dim=4864, n_layers=152, n_heads=152. Scaling only the number of layers, the biggest model has **48,185,937,920** parameters, achieved with dim=4096, n_layers=216, n_heads=32.

**Implementation**: `load_model_no_fsdp.py` \
**Sbatch file**: `sbatch_files/load_model_no_fsdp.sh` \
**Log files**: `logs/load_model_no_fsdp/lsai-453992.out` (scaling all parameters), `logs/load_model_no_fsdp/lsai-454054.out` (scaling only the number of layers)

## Experiment 2: loss ablation with FSDP and without FSDP 
To prove the correctness of FSDP implementation, we fix the seed and train the same model with FSDP (trained on 2 nodes with 4 GPUs each) or without FSDP. Then we compare the loss values parsed from the log files.  

**Results**:
![My Plot](plots/loss_comparison_2025-05-21_01-25-12.png)
```
=== Max Loss Difference ===
Step: 20
Log FSDP Loss: 11.3200
Log NO FSDP Loss: 11.3300
Absolute Difference: 0.0100

=== Mean Metrics ===
Log FSDP:
  tokens_per_sec: 4814.07
  training_tokens_pct: 27.98
  mfu: 6.08
  tflops: 60.09
Log NO FSDP:
  tokens_per_sec: 7515.06
  training_tokens_pct: 23.94
  mfu: 39.16
  tflops: 387.34
```

As we can see, the results are indentical (with the biggest difference in loss values of 0.01) proving the correct implementation of FSDP. 

**Implementation**: `loss_ablation.py` \
**Reproduction**: \
Activate a conda environment:
```bash
$ conda activate 
```

And run: 
```bash
python loss_ablation.py --fsdp-logs=/users/elyulina/scratch/lai-proj/logs/loss_ablation_fsdp/lsai-454149.out --no-fsdp-logs=/users/elyulina/scratch/lai-proj/logs/loss_ablation_no_fsdp/lsai-454162.out
```

## Experiment 3: increasing the model's size with FSDP

trying to make it work and it doesn't :\(

[PyTorch Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html) \
[Ohio Supercomputer Center Tutorial](https://www.osc.edu/resources/getting_started/howto/howto_pytorch_fully_sharded_data_parallel_fsdp) \
[Medium Tutorial](https://medium.com/@kyeg/unlock-multi-gpu-finetuning-secrets-huggingface-models-pytorch-fsdp-explained-a58bab8f510e)








