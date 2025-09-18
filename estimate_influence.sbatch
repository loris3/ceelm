#!/bin/bash
#SBATCH --job-name="[CFE] Influence Estimation Sample"

#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=265GB
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-1:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version

df -h
ulimit -n 65535

mkdir -p /tmp/cache/gradients/ && find cache/gradients/ -type f \
  | sed "s|^cache/gradients/|./|" \
  | xargs -n 100 -P 8 -I{} rsync --relative cache/gradients/{} /tmp/cache/gradients/



python3 estimate_influence.py --base_model allenai/OLMo-2-0425-1B --model_dir ./models/OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42 --train_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample --test_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample  --train_dataset_split=train --test_dataset_split=test


mkdir -p cache/gradients && find /tmp/cache/gradients/ -type f \
  | sed "s|^/tmp/cache/gradients/|./|" \
  | xargs -n 100 -P 8 -I{} rsync --relative /tmp/cache/gradients/{} cache/gradients/

# rm -rf /tmp/cache/gradients/



python3 estimate_influence.py --base_model meta-llama/Llama-3.2-1B --model_dir ./models/Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42 --train_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample --test_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample  --train_dataset_split=train --test_dataset_split=test

mkdir -p cache/gradients && find /tmp/cache/gradients/ -type f \
  | sed "s|^/tmp/cache/gradients/|./|" \
  | xargs -n 100 -P 8 -I{} rsync --relative /tmp/cache/gradients/{} cache/gradients/


python3 estimate_influence.py --base_model Qwen/Qwen-2.5-0.5B --model_dir ./models/Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42 --train_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample --test_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample  --train_dataset_split=train --test_dataset_split=test

mkdir -p cache/gradients && find /tmp/cache/gradients/ -type f \
  | sed "s|^/tmp/cache/gradients/|./|" \
  | xargs -n 100 -P 8 -I{} rsync --relative /tmp/cache/gradients/{} cache/gradients/



# srun   --job-name="[CFE] Influence Estimation Sample"    --container-image=ghcr.io#ghcr.io#loris3/cfe:latest   --nodelist=dgx-h100-em2   --container-mount-home   --mem=256G   --cpus-per-task=64   --gres=gpu:4   --time=1-1:00:00   --container-workdir=/srv/home/users/loriss21cs/cfe   --nodes=1   --pty bash