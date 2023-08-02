#!/bin/bash -l
#$ -N test-job 
#$ -cwd
#$ -l m_mem_free=50G
#$ -l g=1
#$ -j y

module avail
nvidia-smi
nvcc --version


module purge  