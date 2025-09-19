#!/bin/bash
#SBATCH --job-name="[CFE] Scoring"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1
#SBATCH --container-mount-home 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:4
#SBATCH --time=1-1:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h
ulimit -n 65535

python3 score.py


