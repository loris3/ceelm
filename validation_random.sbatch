#!/bin/bash
#SBATCH --job-name="[CFE] Validation Random"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,shelob,galadriel,vader
#SBATCH --container-mount-home
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-19%100  
#SBATCH --partition=p_low
#SBATCH --requeue
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535

export HF_TRANSFER=1
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_MODULES_CACHE=/tmp/hf_cache/modules
export HF_TOKEN=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


SEEDS=(42 10 3 9 6)
K_VALUES=(1 5 10 25)

NUM_SEEDS=${#SEEDS[@]}
NUM_K=${#K_VALUES[@]}

TOTAL_JOBS=$(( NUM_SEEDS * NUM_K ))

# Determine indices for this array job
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_K ))
K_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_K ))

SEED=${SEEDS[$SEED_IDX]}
K=${K_VALUES[$K_IDX]}

echo "Running KRandom validation with seed=${SEED}, k=${K}"
python3 validation.py --explanation_type=KRandom --seed=${SEED} --k=${K}