#!/bin/bash
#SBATCH --job-name="[CFE] Validation of Validation"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1,shelob,galadriel
#SBATCH --container-mount-home
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=1-1:00:00
#SBATCH --nodes=1
#SBATCH --array=0-26%16  
#SBATCH --partition=p_low
#SBATCH --requeue

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
python3 --version
df -h
ulimit -n 65535
export HF_TOKEN=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
export HF_TRANSFER=1
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_MODULES_CACHE=/tmp/hf_cache/modules


LRS=(1e-5 1e-4 1e-3)
EPOCHS=(1 2 5)
REPEATS=(1 2 8)


COMINATIONS=()
for lr in "${LRS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    for repeat in "${REPEATS[@]}"; do
      COMINATIONS+=("$lr $epoch $repeat")
    done
  done
done


PARAMS=(${COMINATIONS[$SLURM_ARRAY_TASK_ID]})
LR=${PARAMS[0]}
EPOCH=${PARAMS[1]}
REPEAT=${PARAMS[2]}

echo "Running with LR=$LR, Epoch=$EPOCH, Repeat=$REPEAT"

python3 validation_experiment.py --lr $LR --epoch $EPOCH --repeat $REPEAT