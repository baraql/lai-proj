# Large Scale AI Engineering Merger: FSDP x Flash Attention 

This repository contains the project for the ETH course Large-Scale AI Engineering. It implements a merger of two features — **Fully Sharded Data Parallel (FSDP)** and **Flash Attention** — for a Transformer model and runs experiments on a multi-GPU cluster.

To view the features separately, see the `feature-fsdp` and `feature-flash-attention` branches, respectively.

## Project Structure
- `sbatch_files/` — contains Slurm job scripts

- `*.py` — training, evaluation, and model code

-  `logs/` — output logs (written per job)

## 🚀 Getting Started 
For convinience, we explicitely provide the environment file `ngc_pt_jan.toml`.
To make it work with the experiments, run the following command:
```
$ mkdir -p ~/.edf/
$ cp ngc_pt_jan.toml ~/.edf/.
```

Then you can just specify `#SBATCH --environment=ngc_pt_jan` in sbatch files. 


## Running Merged Experiments

Please use the `sbatch_files/train_flash_attention_fsdp.sh` file to run merged experiments. 

To control the number of GPUs for FSDP implementation, specify the required number of GPUs via SBATCH directives (it will be equal to `nodes` × `gpu_per_node`).

To control other experiment settings, there are several new flags/arguments you can use:

- `--fused-attention` -- a flag to enable fused attention (from Flash Attention feature)
- `--scaling-factor 10` -- an argument to control the model's scaling factor: 1 represents a very small model, 14 represents the biggest model we can train with FSDP on multiple GPUs (from FSDP feature)
- `--scaling-strategy all` -- an argument to choose between two scaling strategies; the default (recommended) option `all` will scale all parameters simultaneously (from FSDP feature)
- `--set-seed 42` -- sets the seed to the specified value (from FSDP feature)

Finally, to run an experiment, simply execute:
```
sbatch sbatch_files/train_flash_attention_fsdp.sh
```

The corresponding log files will appear in the `logs/train_flash_attention_fsdp/` folder. 

## Running Individual Experiments 
Use `sbatch_files/flash_attention.sh`, `sbatch_files/train_fsdp.sh`, or `sbatch_files/train_no_fsdp.sh` to train a model with only flash attention, only FSDP, or neither.


