#!/bin/bash
#SBATCH --job-name="[CEELM] Validation FacilityLocation"
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-8:00:00
#SBATCH --nodes=1
#SBATCH --array=0-79%90    
#SBATCH --partition=p_low
#SBATCH --requeue

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535
source .env


OTHER_TYPES=("FacilityLocationMostHelpful" "FacilityLocationLeastInfluential" "FacilityLocationMostInfluential" "FacilityLocationMostHarmful")
K_VALUES=(1 25 10 5)
LAMBDA_VALUES=(0.0 0.25 0.5 0.75 1.0)
M_VALUES=(100)


NUM_TYPES=${#OTHER_TYPES[@]}
NUM_K=${#K_VALUES[@]}
NUM_LAMBDA=${#LAMBDA_VALUES[@]}
NUM_M=${#M_VALUES[@]}


TYPE_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_K * NUM_LAMBDA * NUM_M) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_K * NUM_LAMBDA * NUM_M) ))
K_IDX=$(( REMAINDER / (NUM_LAMBDA * NUM_M) ))
REMAINDER2=$(( REMAINDER % (NUM_LAMBDA * NUM_M) ))
LAMBDA_IDX=$(( REMAINDER2 / NUM_M ))
M_IDX=$(( REMAINDER2 % NUM_M ))

EXPLANATION_TYPE=${OTHER_TYPES[$TYPE_IDX]}
K=${K_VALUES[$K_IDX]}
LAMBDA=${LAMBDA_VALUES[$LAMBDA_IDX]}
M=${M_VALUES[$M_IDX]}

echo "Running validation with explanation_type=${EXPLANATION_TYPE}, k=${K}, lambda=${LAMBDA}, m=${M}"

python3 validation.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --lambda_=${LAMBDA} --m=${M}
