#!/bin/bash -l

#SBATCH --account=h_ghazik
#SBATCH --mem=64G
#SBATCH --job-name=p2s               # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gpus=2                     # number of gpus
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=all  # or migs or phys, based on your need

module load python/3.9.6
module load anaconda/3.2023.03
module load cuda/12.1.1

source /usr/local/pkg/anaconda/v3.2023.03/root/etc/profile.d/conda.sh
conda activate py39

nvidia-smi

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
plm_secondary_accelerate.py --deepspeed ds_config_p2s.json'