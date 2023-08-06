#!/bin/bash -l

# must fields: 
#SBATCH --account=h_ghazik     # to specify account name (must field)
#SBATCH --mem=50G           # memory reservation (must field)

# optional fields: 
#SBATCH -J test_job23              # job name
#SBATCH -o _%x%J.out            # Standard output and error log
#SBATCH --gpus=10gb:1         # gpu reservation

# SBATCH -n 1 --ntasks-per-core=1

# set modules
module load anaconda/3.2019.10/default
module load cuda/default
module load python/3.7.3/default

nvidia-smi

# clean loaded modules
module purge  