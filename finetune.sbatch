#!/bin/bash
#SBATCH --job-name="[CFE] Finetuning Sweep"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=1-16:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1  
 

# Print environment info
python3 --version
df -h
export HF_TOKEN=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
export HF_TRANSFER=1

export TOKENIZERS_PARALLELISM=true

python3 -m  wandb agent loriss/cfe_finetuning/tsbm7skf
python3 -m  wandb agent loriss/cfe_finetuning/9pehckf1