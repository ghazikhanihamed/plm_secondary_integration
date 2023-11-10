#!/encs/bin/tcsh

#SBATCH --job-name=plm_p2s
#SBATCH --mem=128G
#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gres=gpu:1
#SBATCH -w speed-43

setenv TMPDIR /nfs/speed-scratch/h_ghazik/tmp
setenv TRANSFORMERS_CACHE /nfs/speed-scratch/h_ghazik/tmp

module load anaconda3/2023.03/default
module load pytorch/1.10.0/GPU/default

conda activate py3.9

python plm_secondary_integration.py

conda deactivate
module purge