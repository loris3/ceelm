#!/bin/bash
WORKDIR="/srv/home/users/loriss21cs/cfe"
EVALDIR="../finetuning_eval"

cd "$WORKDIR/olmes" || { echo "Clone olmes first!"; exit 1; }


tasks=(
    core_9mcqa::olmes
    "mmlu:mc::olmes"
    "olmo_2_generative::olmes"
    "olmo_2_heldout::olmes"
)

models=(
    "allenai/OLMo-2-0425-1B OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42"
    "meta-llama/Llama-3.2-1B Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42" 
    "Qwen/Qwen2.5-0.5B Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42" 
)

baseline_models=(
    "allenai/OLMo-2-0425-1B-SFT"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
)


merge_list=()
for entry in "${models[@]}"; do
    base_model=$(echo "$entry" | cut -d' ' -f1)
    adapter_model=$(echo "$entry" | cut -d' ' -f2)
    merged_model_dir="../models/${adapter_model}-merged"
    merge_list+=("$base_model|$adapter_model|$merged_model_dir")
done

export MERGE_LIST="$(printf "%s\n" "${merge_list[@]}")"

merge_array_length=$(echo "$MERGE_LIST" | wc -l)
merge_script=$(mktemp)
cat > "$merge_script" <<EOL
#!/bin/bash
#SBATCH --job-name="[CFE] Merge"
# #SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx1,dgx-h100-em2,galadriel,shelob
#SBATCH --container-mount-home
#SBATCH --mem=24GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=0-01:00:00
# #SBATCH --container-workdir=$WORKDIR
#SBATCH --array=0-$(($merge_array_length - 1))%4
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_MODULES_CACHE=/tmp/hf_cache/modules
export HF_TOKEN=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

cd $WORKDIR/olmes
if ! grep -qxF 'import multiprocessing as mp' oe_eval/run_eval.py; then
    sed -i '/if __name__ == "__main__":/a import multiprocessing as mp\nmp.set_start_method("spawn", force=True)' oe_eval/run_eval.py
fi

IFS=\$'\n' read -r -d '' -a items <<< "\$MERGE_LIST"
IFS='|' read -r base_model adapter_model merged_model_dir <<< "\${items[\$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "\$merged_model_dir" ]; then
    echo "Merging \$adapter_model"
    python3 ../merge_lora_to_hf.py \
        --base_model "\$base_model" \
        --adapter_dir "../models/\${adapter_model}" \
        --output_dir "\$merged_model_dir"
else
    echo "Already merged: \$merged_model_dir"
fi
EOL

merge_jid=$(sbatch "$merge_script" | awk '{print $4}')
rm "$merge_script"


eval_list=()
for task in "${tasks[@]}"; do
    for entry in "${models[@]}"; do
        adapter_model=$(echo "$entry" | cut -d' ' -f2)
        merged_model_dir="../models/${adapter_model}-merged"
        out_dir="$EVALDIR/$task/${adapter_model}-merged"
        eval_list+=("$task|$merged_model_dir|$out_dir")
    done
    for base_model in "${baseline_models[@]}"; do
        out_dir="$EVALDIR/$task/${base_model}"
        eval_list+=("$task|$base_model|$out_dir")
    done
done

export EVAL_LIST="$(printf "%s\n" "${eval_list[@]}")"

eval_array_length=$(echo "$EVAL_LIST" | wc -l)
eval_script=$(mktemp)
cat > "$eval_script" <<EOL
#!/bin/bash
#SBATCH --job-name="[CFE] Eval cfe:eval"
#SBATCH --nodelist=dgx1,dgx-h100-em2,galadriel,shelob
#SBATCH --container-mount-home
#SBATCH --partition=p_low
#SBATCH --container-image="ghcr.io#loris3/cfe:eval"
#SBATCH --requeue
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --container-workdir=$WORKDIR
#SBATCH --array=0-$(($eval_array_length - 1))%12
export HF_TOKEN=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

cd $WORKDIR/olmes

IFS=\$'\n' read -r -d '' -a items <<< "\$EVAL_LIST"
IFS='|' read -r task model out_dir <<< "\${items[\$SLURM_ARRAY_TASK_ID]}"
model_name=\$(basename "\$model")
echo "Evaluating task=\$task model=\$model out_dir=\$out_dir model_name=\$model_name "

python3 -m oe_eval.launch \
    --model "\$model" \
    --num-workers=1 \
    --gpus=1 \
    --task "\$task" \
    --output-dir "\$out_dir"
EOL


sbatch --dependency=afterok:$merge_jid "$eval_script"

