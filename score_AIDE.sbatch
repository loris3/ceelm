#!/bin/bash
#SBATCH --job-name="[CFE] Scoring AIDE"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,shelob,galadriel,vader
#SBATCH --container-mount-home
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-3%100  
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

DIVINE_TYPES=("AIDE")
K_VALUES=(1 5 10 25)
M_VALUES=(100)


NUM_K=${#K_VALUES[@]}
NUM_M=${#M_VALUES[@]}

TOTAL_JOBS=$(( 1 * NUM_K * NUM_M ))

# Determine indices for this array job
TYPE_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_K * NUM_M) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_K * NUM_M) ))
K_IDX=$(( REMAINDER / NUM_M ))
M_IDX=$(( REMAINDER % NUM_M ))

EXPLANATION_TYPE=${DIVINE_TYPES[$TYPE_IDX]}
K=${K_VALUES[$K_IDX]}
M=${M_VALUES[$M_IDX]}

echo "Running AIDE scoring with explanation_type=${EXPLANATION_TYPE}, k=${K}, m=${M}"
python3 score.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --m=${M}