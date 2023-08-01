#!/bin/bash -l

# must fields: 
#SBATCH --account=h_ghazik     # to specify account name (must field)
#SBATCH --mem=128G           # memory reservation (must field)

# optional fields: 
#SBATCH -J plm_sec              # job name
#SBATCH -o _%x%J.out            # Standard output and error log
#SBATCH --gpus=10gb:4         # gpu reservation
#SBATCH --mail-user=h_ghazik

# SBATCH -n 1 --ntasks-per-core=1

sleep 1m
module avail
nvidia-smi
nvcc --version

# module load gcc/10.1.0/default
# module load cuda/11.4/default

# source ~/venv_secondary/bin/activate

# accelerate launch --multi_gpu plm_secondary_deepspeed.py

# clean loaded modules
module purge