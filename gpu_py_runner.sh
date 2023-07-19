#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=200G,g=4

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

source ~/python_venv/bin/activate

deepspeed --num_gpus=4 plm_secondary_integration.py 

# python plm_secondary_integration.py 

deactivate

