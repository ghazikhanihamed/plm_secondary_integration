#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G
#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=10gb:4
#SBATCH -w virya3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamed.ghazikhani@gmail.com

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/11.7.0
source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

accelerate launch --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml --multi_gpu binary_classification_ionchannel_task.py

conda deactivate
module purge
