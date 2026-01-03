#!/bin/bash
#SBATCH --job-name="[CEELM] Notebook Run"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --array=0-4

mkdir -p ./out


source .env
ulimit -n 65535



python3 --version
df -h

NOTEBOOKS=(
  "case_study.ipynb"
  "finetuning_eval.ipynb"
  "progress.ipynb"
  "scoring.ipynb"
  "validation.ipynb"
)

NB=${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}
OUT="./out/${NB%.ipynb}.ipynb"

echo "Running notebook: $NB"
echo "Output notebook: $OUT"


python3.10 -m jupyter nbconvert \
  --to notebook \
  --execute "$NB" \
  --output "$OUT"
