#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G

#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=10gb:3

# SBATCH -n 1 --ntasks-per-core=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/11.7.0
source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

export PATH="$PATH:/home/h_ghazik/.local/bin"
export ACCELERATE_CONFIG=/home/h_ghazik/accelerate_config/default_config.yaml

accelerate launch --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml --multi_gpu plm_secondary_deepspeed.py

conda deactivate

module purge


# srun --account=h_ghazik -w virya3 --mem=10GB --gpus=10gb:4 --pty /bin/zsh