#!/bin/bash
#SBATCH --job-name="[CFE] Pre-tokenization before finetuning"

#SBATCH --container-image=ghcr.io#ghcr.io#loris3/cfe:latest 
#SBATCH --container-mount-home 
#SBATCH --mem=300G
#SBATCH --cpus-per-task=64  
#SBATCH --nodelist=dgx1,dgx-h100-em2
#SBATCH --gres=gpu:0
#SBATCH --time=0-1:00:00
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version
python3  tokenize_dataset_ft.py  --base_model allenai/OLMo-2-0425-1B   --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
python3  tokenize_dataset_ft.py  --base_model meta-llama/Llama-3.2-1B   --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
python3  tokenize_dataset_ft.py  --base_model Qwen/Qwen2.5-0.5B  --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
