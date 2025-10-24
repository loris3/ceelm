#!/bin/bash
#SBATCH --job-name="[CFE] Validation"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,galadriel,shelob,vader
#SBATCH --container-mount-home
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-11%10  
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
OTHER_TYPES=("FacilityLocationLeastInfluential" "FacilityLocationMostInfluential" "FacilityLocationMostHelpful" "FacilityLocationMostHarmful" "Self" "TopKMostInfluential" "TopKLeastInfluential" "TopKMostHelpful" "TopKMostHarmful")
K_VALUES=(1 5 10 25)
LAMBDA_VALUES=(0.0 0.5 1)
M_VALUES=(100)

NUM_SEEDS=${#SEEDS[@]}
NUM_TYPES=${#OTHER_TYPES[@]}
NUM_K=${#K_VALUES[@]}
NUM_LAMBDA=${#LAMBDA_VALUES[@]}
NUM_M=${#M_VALUES[@]}

NUM_FACILITY_TYPES=4

# Total jobs
TOTAL_KRANDOM=$(( NUM_SEEDS * NUM_K ))
TOTAL_FACILITY=$(( NUM_FACILITY_TYPES * NUM_K * NUM_LAMBDA * NUM_M ))
TOTAL_OTHER=$(( (NUM_TYPES - NUM_FACILITY_TYPES) * NUM_K * NUM_LAMBDA ))
TOTAL_COMBINATIONS=$(( TOTAL_KRANDOM + TOTAL_FACILITY + TOTAL_OTHER ))

# Determine which block the task belongs to
if [ $SLURM_ARRAY_TASK_ID -lt $TOTAL_KRANDOM ]; then
    # KRandom
    ARRAY_IDX=$SLURM_ARRAY_TASK_ID
    SEED_IDX=$(( ARRAY_IDX / NUM_K ))
    K_IDX=$(( ARRAY_IDX % NUM_K ))
    SEED=${SEEDS[$SEED_IDX]}
    K=${K_VALUES[$K_IDX]}
    echo "Running KRandom with seed=${SEED}, k=${K}"
    python3 validation.py --explanation_type=KRandom --seed=${SEED} --k=${K}

elif [ $SLURM_ARRAY_TASK_ID -lt $(( TOTAL_KRANDOM + TOTAL_FACILITY )) ]; then
    # FacilityLocation* types
    ARRAY_IDX=$(( SLURM_ARRAY_TASK_ID - TOTAL_KRANDOM ))
    TYPE_IDX=$(( ARRAY_IDX / (NUM_K * NUM_LAMBDA * NUM_M) ))
    REMAINDER=$(( ARRAY_IDX % (NUM_K * NUM_LAMBDA * NUM_M) ))
    K_IDX=$(( REMAINDER / (NUM_LAMBDA * NUM_M) ))
    REM2=$(( REMAINDER % (NUM_LAMBDA * NUM_M) ))
    LAMBDA_IDX=$(( REM2 / NUM_M ))
    M_IDX=$(( REM2 % NUM_M ))

    EXPLANATION_TYPE=${OTHER_TYPES[$TYPE_IDX]}
    K=${K_VALUES[$K_IDX]}
    LAMBDA=${LAMBDA_VALUES[$LAMBDA_IDX]}
    M=${M_VALUES[$M_IDX]}

    echo "Running Facility scoring with explanation_type=${EXPLANATION_TYPE}, k=${K}, lambda=${LAMBDA}, m=${M}"
    python3 validation.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --lambda_=${LAMBDA} --m=${M}

else
    # Non-Facility types
    ARRAY_IDX=$(( SLURM_ARRAY_TASK_ID - TOTAL_KRANDOM - TOTAL_FACILITY ))
    TYPE_IDX=$(( ARRAY_IDX / (NUM_K * NUM_LAMBDA) + NUM_FACILITY_TYPES ))
    REMAINDER=$(( ARRAY_IDX % (NUM_K * NUM_LAMBDA) ))
    K_IDX=$(( REMAINDER / NUM_LAMBDA ))
    LAMBDA_IDX=$(( REMAINDER % NUM_LAMBDA ))

    EXPLANATION_TYPE=${OTHER_TYPES[$TYPE_IDX]}
    K=${K_VALUES[$K_IDX]}
    LAMBDA=${LAMBDA_VALUES[$LAMBDA_IDX]}

    echo "Running scoring with explanation_type=${EXPLANATION_TYPE}, k=${K}, lambda=${LAMBDA}"
    python3 validation.py --explanation_type=${EXPLANATION_TYPE} --k=${K} --lambda_=${LAMBDA}
fi
