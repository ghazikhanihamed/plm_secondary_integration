#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=128G,g=4

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6/default
module load anaconda/3.2022.10/default
module load gcc/10.1.0/default
module load cuda/11.4/default

source /usr/local/pkg/Anaconda/Anaconda3.2022.10/root/etc/profile.d/conda.sh
conda activate py3.9

accelerate launch --multi_gpu plm_secondary_deepspeed.py

conda deactivate
module purge