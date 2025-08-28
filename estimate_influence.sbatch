#!/bin/bash
#SBATCH --job-name="[CFE] Influence Estimation"

#SBATCH --container-image=loris3/cfe:latest 
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-10:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h



python3 estimate_influence.py --base_model EleutherAI/pythia-31m --model_dir ./models/pythia-31m_tulu-v2-sft-mixture_train --train_dataset allenai/tulu-v2-sft-mixture --test_dataset allenai/tulu-v2-sft-mixture  --train_dataset_split=train[0:2000] --test_dataset_split=train[0:100]

python3 estimate_influence.py --base_model allenai/OLMo-2-0425-1B --model_dir ./models/OLMo-2-0425-1B_tulu-v2-sft-mixture_train --train_dataset allenai/tulu-v2-sft-mixture --test_dataset allenai/tulu-v2-sft-mixture --train_dataset_split=train[0:2000] --test_dataset_split=train[0:100]
