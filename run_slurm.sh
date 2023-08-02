#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G

#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=10gb:2

# SBATCH -n 1 --ntasks-per-core=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load anaconda/3.2023.03
module load cuda/12.0.0
module load python/3.9.6 

gcc --version
nvcc --version

source ~/venv_secondary/bin/activate

accelerate launch --multi_gpu plm_secondary_deepspeed.py

deactivate

module purge