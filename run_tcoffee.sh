#!/encs/bin/tcsh

#SBATCH --job-name=tcoffee
#SBATCH --mem=128G
#SBATCH -o _%x%J.out
#SBATCH --cpus-per-task=8


setenv TMPDIR /nfs/speed-scratch/h_ghazik/tmp

module load anaconda3/2023.03/default

conda activate tcoffee_env_py36

python align_sequences.py

conda deactivate
module purge

