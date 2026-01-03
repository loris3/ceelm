#!/bin/bash
#SBATCH --job-name="[CEELM] Selection AIDE"
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-8:00:00
#SBATCH --nodes=1
#SBATCH --array=0-3%90    
#SBATCH --partition=p_low
#SBATCH --requeue

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535
source .env

OTHER_TYPES=("AIDE")
K_VALUES=(10 5 1 25)
M_VALUES=(100)

NUM_TYPES=${#OTHER_TYPES[@]}
NUM_K=${#K_VALUES[@]}
NUM_M=${#M_VALUES[@]}

# Compute indices for this array task
TYPE_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_K * NUM_M) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_K * NUM_M) ))
K_IDX=$(( REMAINDER / NUM_M ))
M_IDX=$(( REMAINDER % NUM_M ))

EXPLANATION_TYPE=${OTHER_TYPES[$TYPE_IDX]}
K=${K_VALUES[$K_IDX]}
M=${M_VALUES[$M_IDX]}

echo "Running selection with explanation_type=${EXPLANATION_TYPE}, k=${K}, m=${M}"
python3 selection.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --m=${M}