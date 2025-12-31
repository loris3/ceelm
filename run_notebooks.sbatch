#!/bin/bash
#SBATCH --job-name="[CEELM] Notebook Run"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --container-mount-home
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe


#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00

#SBATCH --array=0-4


export HF_HOME="/mnt/nlp-data/home/users/loriss21cs/.hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=cba4442ed0417ba698aa1b18b17ee238f46945af
ulimit -n 65535



python3 --version
df -h

# --------------------
# Notebook selection
# --------------------
NOTEBOOKS=(
  "case_study.ipynb"
  "finetuning_eval.ipynb"
  "progress.ipynb"
  "scoring.ipynb"
  "validation.ipynb"
)

NB=${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}
OUT="${NB%.ipynb}_executed.ipynb"

echo "Running notebook: $NB"
echo "Output notebook: $OUT"

# --------------------
# Execute notebook
# --------------------
python3.10 -m jupyter nbconvert \
  --to notebook \
  --execute "$NB" \
  --output "$OUT"
