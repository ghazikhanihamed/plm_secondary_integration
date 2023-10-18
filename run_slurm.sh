#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128GB
#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=4
#SBATCH -w virya3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamed.ghazikhani@gmail.com

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/12.1.1
conda activate py39

nvidia-smi

accelerate launch plm_secondary_accelerate.py

conda deactivate
module purge


# salloc  -A h_ghazik -t 24:00:00 --mem=64G --gpus=1 
# ~/plm_secondary_integration/ds_config_p2s.json
# accelerate launch --multi_gpu plm_secondary_accelerate.py
# --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml