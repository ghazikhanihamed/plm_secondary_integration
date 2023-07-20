#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=200G,g=4

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/venv_secondary/bin/activate

deepspeed --num_gpus=4 plm_secondary_integration.py 

deactivate

