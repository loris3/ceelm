#!/bin/bash
#SBATCH --job-name="[CFE] Finetuning full tulu"

#SBATCH --container-image=loris3/cfe:latest 
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:2
#SBATCH --time=1-10:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h


# torchrun --nnodes 1  
python3 -m accelerate.commands.launch finetune.py \
  --base_model EleutherAI/pythia-31m \
  --train_dataset allenai/tulu-v2-sft-mixture \
  --train_split train \
  --output_dir ./models/pythia-31m_tulu-v2-sft-mixture_train \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --logging_steps 10 \
  --seed 42 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --bf16 True \
  --report_to wandb

python3 -m accelerate.commands.launch finetune.py \
  --base_model allenai/OLMo-2-0425-1B \
  --train_dataset allenai/tulu-v2-sft-mixture \
  --train_split train[0%:10%] \
  --output_dir ./models/OLMo-2-0425-1B_tulu-v2-sft-mixture_train \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --logging_steps 10 \
  --seed 42 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --bf16 True \
  --report_to wandb
