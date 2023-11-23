#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G
#SBATCH -J save_emb
#SBATCH -o _%x%J.out
#SBATCH --gpus=1

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/12.1.1

source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

nvidia-smi

python save_embeddings.py

conda deactivate
module purge


# salloc  -A h_ghazik -t 24:00:00 --mem=64G --gpus=1
# ~/plm_secondary_integration/ds_config_p2s.json
# accelerate launch --multi_gpu plm_secondary_accelerate.py
# --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml

# #SBATCH -w virya1

# speed
# module load python/3.9.1/default
# module load anaconda3/2023.03/default
# module load cuda/11.8/default

# source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.csh
# conda activate py3.9
