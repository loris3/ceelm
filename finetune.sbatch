#!/bin/bash
#SBATCH --job-name="[CFE] Finetuning Sweep OLMo Llama Qwen"

#SBATCH --container-image=ghcr.io#ghcr.io#loris3/cfe:latest 
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:4
#SBATCH --time=1-16:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h
python3 -m   wandb agent loriss/cfe_finetuning/bdn9fl7r
