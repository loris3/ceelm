#!/bin/bash
#SBATCH --job-name="[CFE] Scoring Sample"
#SBATCH --dependency=afterok:237475
#SBATCH --container-image=ghcr.io#ghcr.io#loris3/cfe:latest 
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-1:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h
ulimit -n 65535

python3 score.py


