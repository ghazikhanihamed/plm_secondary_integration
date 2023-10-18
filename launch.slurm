#!/bin/bash -l

#SBATCH -A=h_ghazik
#SBATCH --mem=64G
#SBATCH --job-name=p2s        # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
# #SBATCH --cpus-per-task=10         # number of cores per tasks
#SBATCH --gpus=4                     # number of gpus
# #SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
plm_secondary_accelerate.py --deepspeed ds_config_p2s.json'