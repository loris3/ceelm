#!/bin/bash
#SBATCH --job-name="[CFE] Influence Estimation"
#SBATCH --array=0-2%1
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=725G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-7:59:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1

python3 --version
export HF_TOKEN=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
df -h
ulimit -n 65535
pip show datasets

MODELS=(
  "allenai/OLMo-2-0425-1B ./models/OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
  "meta-llama/Llama-3.2-1B ./models/Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
  "Qwen/Qwen-2.5-0.5B ./models/Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
)

# Extract model + directory based on task ID
TASK="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASE_MODEL=$(echo $TASK | awk '{print $1}')
MODEL_DIR=$(echo $TASK | awk '{print $2}')

echo "Running task $SLURM_ARRAY_TASK_ID with base_model=$BASE_MODEL, model_dir=$MODEL_DIR"

python3 estimate_influence.py \
  --base_model "$BASE_MODEL" \
  --model_dir "$MODEL_DIR" \
  --train_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample \
  --test_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample \
  --train_dataset_split=train \
  --test_dataset_split=test

rm -rf /tmp/cache/gradients
