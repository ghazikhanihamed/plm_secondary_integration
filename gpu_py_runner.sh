#!/bin/bash -l

#$ -N plm_sec
#$ -cwd
#$ -m bea
#$ -l m_mem_free=128G,g=4
#$ -l hostname=virya


module load python/3.9.6/default
module load anaconda/3.2022.10/default
module load gcc/10.1.0/default
module load cuda/11.4/default

source /usr/local/pkg/Anaconda/Anaconda3.2022.10/root/etc/profile.d/conda.sh
conda activate py3.9

export TMPDIR=~/tmp
export TRANSFORMERS_CACHE=~/tmp

# python feature_extraction_test_pipeline.py
# deepspeed --num_gpus 2 plm_secondary_deepspeed.py
# accelerate launch --multi_gpu preliminary_test.py
# accelerate launch --multi_gpu plm_secondary_accelerate.py

accelerate launch --config_file /home/h_ghazik/.cache/huggingface/accelerate/default_config.yaml --multi_gpu plm_secondary_accelerate.py


conda deactivate
module purge
