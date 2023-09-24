#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=128G
#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gpus=10gb:4
#SBATCH -w virya4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hamed.ghazikhani@gmail.com

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/11.7.0
source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

nvidia-smi

accelerate launch --multi_gpu plm_secondary_accelerate.py

conda deactivate
module purge


# srun --account=h_ghazik -w virya4 -t 24:00:00 --mem=128G --gpus=10gb:4 --pty /bin/zsh
# ~/plm_secondary_integration/ds_config_p2s.json
# accelerate launch --multi_gpu plm_secondary_accelerate.py
# --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml