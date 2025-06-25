#!/bin/bash -l
#SBATCH --job-name="vladbuff_train"  # Job name
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # Adjust CPU cores as needed
#SBATCH --mem=48G                 # Adjust memory as needed
#SABTCH --mem-per-gpu=32G
#SBATCH --time=12:00:00           # Job time limit (HH:MM:SS)
#SBATCH --output=/mnt/beegfs/xliu0001/logs/%x_%j.out   # Output log file
#SBATCH --error=/mnt/beegfs/xliu0001/logs/%x_%j.err    # Error log file
module use /share/apps/eb/modules/all/
# Load required modules (if any)
module load CUDA/11.8.0
module load make

WORKDIR="/mnt/beegfs/xliu0001/vladbuff"

cd $WORKDIR

# Run the job using singularity and CUDA
PATH=$PATH:"/share/apps/singularity/bin" make sing-train
# pixi run train-xfeat
# nvidia-smi
