#!/bin/bash -l
#SBATCH --job-name="bash"  # Job name
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4              # Request 1 GPU
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=64G                 # Adjust memory as needed
#SABTCH --mem-per-gpu=32G
#SBATCH --time=48:00:00           # Job time limit (HH:MM:SS)
#SBATCH --output=/mnt/beegfs/xliu0001/logs/%x_%j.out   # Output log file
#SBATCH --error=/mnt/beegfs/xliu0001/logs/%x_%j.err    # Error log file
module use /share/apps/eb/modules/all/
# Load required modules (if any)
module load CUDA/12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
module load make

WORKDIR="/mnt/beegfs/xliu0001/vladbuff"

cd $WORKDIR

pixi install
# Run the job using singularity and CUDA
# PATH=$PATH:"/share/apps/singularity/bin" make sing-train
srun pixi run train-xfeat
# nvidia-smi
