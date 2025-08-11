#!/bin/bash
#SBATCH --job-name="[CFE] Finetuning Olmo2-1B math"

#SBATCH --container-image=loris3/cfe:latest 
# #SBATCH --container-image=/srv/home/users/loriss21cs/cfe/cfe.sqsh


#SBATCH --nodelist=dgx-h100-em2
# # SBATCH --container-save=/srv/home/users/loriss21cs/cfe/cfe.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:2
#SBATCH --time=1-10:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1

export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=cba4442ed0417ba698aa1b18b17ee238f46945af
export WANDB_PROJECT=cfe
export HF_TOKEN=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
python3 --version

df -h

# python3 trak_test.py --out ./
torchrun --nnodes 1 --standalone -m coreset_trainer.train --ratio=0.25
torchrun --nnodes 1 --standalone -m coreset_trainer.train --ratio=0.5
torchrun --nnodes 1 --standalone -m coreset_trainer.train --ratio=1.0
