#!/bin/bash
#SBATCH --job-name="[CEELM] Scoring Self"
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH --requeue
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535

source .env


EXPLANATION_TYPE="Self"

echo "Running Self scoring with explanation_type=${EXPLANATION_TYPE}"
python3 score.py --explanation_type=${EXPLANATION_TYPE}