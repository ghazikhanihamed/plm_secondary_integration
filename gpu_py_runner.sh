#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=128G,g=3

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

deepspeed --num_gpus=3 plm_secondary_integration.py 

deactivate

