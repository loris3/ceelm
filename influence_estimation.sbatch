#!/bin/bash
#SBATCH --job-name="[CFE] Influence Estimation LESS Olmo2-1B math"

#SBATCH --container-image=loris3/cfe:latest 
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:1
#SBATCH --time=1-10:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h

torchrun --nnodes 1 run_less.py