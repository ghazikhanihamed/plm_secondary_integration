#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=512G,g=8

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load python/3.9.6/default
module load anaconda/3.2022.10/default
module load gcc/10.1.0/default
module load cuda/11.4/default

source /usr/local/pkg/Anaconda/Anaconda3.2022.10/root/etc/profile.d/conda.sh
conda activate py3.9

# python feature_extraction_test_pipeline.py
deepspeed --num_gpus 8 finetune_plm_secondary.py
# accelerate launch --multi_gpu preliminary_test.py


conda deactivate
module purge