#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G

#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=10gb:1

# SBATCH -n 1 --ntasks-per-core=1

module avail
nvidia-smi
nvcc --version

module purge