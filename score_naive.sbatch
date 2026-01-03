#!/bin/bash
#SBATCH --job-name="[CEELM] Scoring Naive"
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-15%100  
#SBATCH --partition=p_low
#SBATCH --requeue
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535
source .env


TOPK_TYPES=("TopKMostInfluential" "TopKLeastInfluential" "TopKMostHelpful" "TopKMostHarmful")
K_VALUES=(1 5 10 25)
M_VALUES=(100)

NUM_TYPES=${#TOPK_TYPES[@]}
NUM_K=${#K_VALUES[@]}
NUM_M=${#M_VALUES[@]}

TOTAL_JOBS=$(( NUM_TYPES * NUM_K * NUM_M ))

# Determine indices for this array job
TYPE_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_K * NUM_M) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_K * NUM_M) ))
K_IDX=$(( REMAINDER / NUM_M ))
M_IDX=$(( REMAINDER % NUM_M ))

EXPLANATION_TYPE=${TOPK_TYPES[$TYPE_IDX]}
K=${K_VALUES[$K_IDX]}
M=${M_VALUES[$M_IDX]}

echo "Running TopK scoring with explanation_type=${EXPLANATION_TYPE}, k=${K}, m=${M}"
python3 score.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --m=${M}