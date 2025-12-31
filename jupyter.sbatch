#!/bin/bash
#SBATCH --job-name="[CEELM] Jupyter Lab Server"

#SBATCH --container-image="ghcr.io#loris3/cfe:latest"



#SBATCH --nodelist=shelob
# # dgx-h100-em2
#SBATCH --container-mount-home 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:0
#SBATCH --time=0-10:00:00
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --nodes=1
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
# #SBATCH --partition=p_datamining
#SBATCH --container-writable
export HF_HOME="/mnt/nlp-data/home/users/loriss21cs/.hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=cba4442ed0417ba698aa1b18b17ee238f46945af
ulimit -n 65535

pip install ipykernel==6.30.1

python3 --version

df -h
echo "http://localhost:2203/lab?token=gzr8ohuifnkjp90o3lewhuiopwelkd"
python3.10 -m jupyterlab --NotebookApp.token='gzr8ohuifnkjp90o3lewhuiopwelkd' --port=2203 --allow-root
