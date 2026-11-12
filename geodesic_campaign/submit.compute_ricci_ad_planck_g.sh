#!/bin/bash -l

#SBATCH --job-name=compute_ricci_ad_planck_g
#SBATCH --nodes=1
#SBATCH --partition=apollo
#SBATCH --nodelist=apollo02
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=out_err_files/compute_ricci_ad_planck_g_%j.out
#SBATCH --error=out_err_files/compute_ricci_ad_planck_g_%j.err

# Load the Conda environment
conda activate torch_cuda_pyro

python -u compute_ricci_ad_planck_g.py
