#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=128G,g=4

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

module load gcc/10.1.0/default
module load cuda/11.4/default

source ~/venv_secondary/bin/activate

accelerate launch --multi_gpu plm_secondary_deepspeed.py
# deepspeed --num_gpus=6 plm_secondary_deepspeed.py
# python plm_secondary_nodeepspeed.py

deactivate

