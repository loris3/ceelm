#!/bin/bash
#SBATCH --job-name="[CEELM] Scoring FacilityLocation"
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-79%100  
#SBATCH --partition=p_low
#SBATCH --requeue
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535

source .env



FACILITY_TYPES=("FacilityLocationLeastInfluential" "FacilityLocationMostInfluential" "FacilityLocationMostHelpful" "FacilityLocationMostHarmful")
K_VALUES=(1 5 10 25)
LAMBDA_VALUES=(0.0 0.25 0.5 0.75 1.0)
M_VALUES=(100)

NUM_TYPES=${#FACILITY_TYPES[@]}
NUM_K=${#K_VALUES[@]}
NUM_LAMBDA=${#LAMBDA_VALUES[@]}
NUM_M=${#M_VALUES[@]}

TOTAL_JOBS=$(( NUM_TYPES * NUM_K * NUM_LAMBDA * NUM_M ))

# Determine indices for this array job
TYPE_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_K * NUM_LAMBDA * NUM_M) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_K * NUM_LAMBDA * NUM_M) ))
K_IDX=$(( REMAINDER / (NUM_LAMBDA * NUM_M) ))
REM2=$(( REMAINDER % (NUM_LAMBDA * NUM_M) ))
LAMBDA_IDX=$(( REM2 / NUM_M ))
M_IDX=$(( REM2 % NUM_M ))

EXPLANATION_TYPE=${FACILITY_TYPES[$TYPE_IDX]}
K=${K_VALUES[$K_IDX]}
LAMBDA=${LAMBDA_VALUES[$LAMBDA_IDX]}
M=${M_VALUES[$M_IDX]}

echo "Running Facility scoring with explanation_type=${EXPLANATION_TYPE}, k=${K}, lambda=${LAMBDA}, m=${M}"
python3 score.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --lambda_=${LAMBDA} --m=${M}