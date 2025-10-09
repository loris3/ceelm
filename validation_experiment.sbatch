#!/bin/bash
#SBATCH --job-name="[CFE] Validation Experiment"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,shelob,galadriel
#SBATCH --container-mount-home
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH --requeue

python3 --version
df -h
ulimit -n 65535
export HF_TOKEN=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
export HF_TRANSFER=1
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_MODULES_CACHE=/tmp/hf_cache/modules

python3 validation_experiment.py 

