#!/bin/bash

#SBATCH -A a-large-sc
#SBATCH --job-name=explore_model_params
#SBATCH --output=logs/explore_model_params_%j.out  # Log output to a 'logs' subdirectory
#SBATCH --error=logs/explore_model_params_%j.err   # Log errors to a 'logs' subdirectory
#SBATCH --nodes=1                    # We need only one node
#SBATCH --ntasks-per-node=1          # One task (our Python script)
#SBATCH --cpus-per-task=4            # Request a few CPUs
#SBATCH --mem=64G                    # Request 64GB of RAM (as step 5/7 mentions "significant memory")
#SBATCH --time=00:20:00              # 20 minutes should be ample time

# --- Cluster-Specific Directives ---
# You MIGHT need to add cluster-specific directives here.
# For example, partition, account, QoS, etc.
# Check the `submit-llama3.sh` script (from step 6/7) for examples relevant to your cluster (e.g., clariden).
# E.g., #SBATCH --partition=your_cpu_partition  OR  #SBATCH --partition=your_gpu_partition if model.py needs GPU
# E.g., #SBATCH --account=your_account

# --- Setup ---
# Project directory (ensure $USER is correctly interpreted or hardcode your username)
PROJECT_DIR="/iopsstor/scratch/cscs/$USER/assignment-2"
PYTHON_SCRIPT_NAME="explore_model.py"
PYTHON_SCRIPT_PATH="$PROJECT_DIR/$PYTHON_SCRIPT_NAME"
MODEL_DEFINITION_PATH="$PROJECT_DIR/model.py"

# Create a logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

echo "----------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Project Directory: $PROJECT_DIR"
echo "Python Script: $PYTHON_SCRIPT_PATH"
echo "----------------------------------------------------"

# Change to the project directory
cd "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to $PROJECT_DIR"
    exit 1
fi

# --- Activate Conda Environment ---
# IMPORTANT: Replace <your_conda_env_name> with the actual name of your conda environment.
# The assignment mentions `conda activate` after `pip install torch transformers pyarrow datasets`.
# Use the name of the environment where you installed these packages.
CONDA_ENV_NAME="base" # <--- !!! REPLACE THIS !!!

echo "Activating conda environment: $CONDA_ENV_NAME..."
# Initialize Conda for bash shell
eval "$(conda shell.bash hook)"
# Or, you might use: source $CONDA_PREFIX/etc/profile.d/conda.sh

conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    echo "Please check the environment name and your conda setup."
    exit 1
fi
echo "Conda environment activated: $(which python)"
echo "Installed packages (transformers, torch):"
pip list | grep -E "transformers|torch"


# --- Pre-run Checks ---
# Ensure model.py is in the PROJECT_DIR (copied in step 5/7)
if [ ! -f "$MODEL_DEFINITION_PATH" ]; then
    echo "Error: Model definition file '$MODEL_DEFINITION_PATH' not found."
    echo "Please ensure you have copied 'model.py' to '$PROJECT_DIR' as per step 5/7."
    exit 1
fi
echo "'model.py' found."

# Ensure the Python script for step 5 exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_PATH' not found."
    echo "Please create it with the code to load the model and print parameters."
    exit 1
fi
echo "Python script '$PYTHON_SCRIPT_NAME' found."

# --- Execute the Python Script ---
echo -e "\nRunning Python script: $PYTHON_SCRIPT_NAME..."
# The `numactl` command from step 7/7 (nsys profiling) might be useful for optimizing
# memory/CPU affinity on NUMA architectures for performance-critical tasks.
# For simply loading a model and counting parameters, it's likely not essential.
# If needed: numactl --membind=0-3 python3 "$PYTHON_SCRIPT_NAME"
python3 "$PYTHON_SCRIPT_NAME"

SCRIPT_EXIT_CODE=$?
echo -e "\nPython script finished with exit code: $SCRIPT_EXIT_CODE"
echo "Job finished at $(date)"
echo "----------------------------------------------------"

exit $SCRIPT_EXIT_CODE
