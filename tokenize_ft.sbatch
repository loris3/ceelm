#!/bin/bash
#SBATCH --job-name="[CEELM] Pre-tokenization before finetuning"
#SBATCH --mem=300G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:0
#SBATCH --time=0-1:00:00
#SBATCH --nodes=1
source .env

python3 --version
python3  tokenize_dataset_ft.py  --base_model allenai/OLMo-2-0425-1B   --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
python3  tokenize_dataset_ft.py  --base_model meta-llama/Llama-3.2-1B   --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
python3  tokenize_dataset_ft.py  --base_model Qwen/Qwen2.5-0.5B  --train_dataset allenai/tulu-3-sft-olmo-2-mixture-0225   --train_split train   
