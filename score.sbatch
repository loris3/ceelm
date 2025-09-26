#!/bin/bash
#SBATCH --job-name="[CFE] Scoring"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1,shelob
#SBATCH --container-mount-home
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-9%6  
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


SEEDS=(42 10 3 9 6)


OTHER_TYPES=("Self" "TopKMostInfluential" "TopKLeastInfluential" "TopKMostHelpful" "TopKMostHarmful")


if [ $SLURM_ARRAY_TASK_ID -lt ${#SEEDS[@]} ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
    echo "Running KRandom with seed=${SEED}"
    python3 score.py --explanation_type=KRandom --seed=${SEED}
else
    TYPE_IDX=$(( SLURM_ARRAY_TASK_ID - ${#SEEDS[@]} ))
    EXPLANATION_TYPE=${OTHER_TYPES[$TYPE_IDX]}
    echo "Running scoring with explanation_type=${EXPLANATION_TYPE}"
    python3 score.py --explanation_type=${EXPLANATION_TYPE}
fi
