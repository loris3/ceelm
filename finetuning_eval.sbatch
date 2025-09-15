#!/bin/bash
#SBATCH --job-name="[CFE] Finetuning Eval"

#SBATCH --container-image=loris3/cfe:latest 
#SBATCH --nodelist=dgx1
#SBATCH --container-mount-home 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:4
#SBATCH --time=1-10:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1


python3 --version
cd olmes


# patch mp spawn method
sed -i '/if __name__ == "__main__":/a \
    import multiprocessing as mp\n\
    mp.set_start_method("spawn", force=True)' oe_eval/run_eval.py



models=(
    # "EleutherAI/pythia-31m pythia-31m_tulu-v2-sft-mixture_train"
    "meta-llama/Llama-3.2-1B Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
    "Qwen/Qwen-2.5-0.5B Qwen2.5-0.5B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
    "allenai/OLMo-2-0425-1B OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr1e-05_seed42"
)

for entry in "${models[@]}"; do
    base_model=$(echo $entry | cut -d' ' -f1)
    adapter_model=$(echo $entry | cut -d' ' -f2)

    echo "Merging $adapter_model"
    python3 ../merge_lora_to_hf.py \
        --base_model "$base_model" \
        --adapter_dir "../models/$adapter_model" \
        --output_dir "../models/${adapter_model}-merged"

    echo "Eval for $adapter_model"
    python3 -m oe_eval.launch \
        --model "../models/${adapter_model}-merged" \
        --num-workers=4 \
        --gpus=4 \
        --task olmo_2_heldout::olmes \
        --output-dir "../finetuning_eval/${adapter_model}-merged"
done