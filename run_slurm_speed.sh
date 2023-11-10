#!/encs/bin/tcsh

#SBATCH --job-name=plm_p2s
#SBATCH --mem=128G
#SBATCH -J plm_sec
#SBATCH -o _%x%J.out
#SBATCH --gres=gpu:1
#SBATCH -w speed-43
#SBATCH --partition=ep


setenv TMPDIR /nfs/speed-scratch/h_ghazik/tmp
setenv TRANSFORMERS_CACHE /nfs/speed-scratch/h_ghazik/tmp

module load python/3.9.1/default
module load anaconda3/2023.03/default
module load pytorch/1.10.0/GPU/default

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.csh
conda activate py39

python plm_secondary_integration.py

conda deactivate
module purge