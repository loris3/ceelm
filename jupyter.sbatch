#!/bin/bash
#SBATCH --job-name="[CFE] Jupyter Lab Server"

#SBATCH --container-image=loris3/cfe:latest 
# #SBATCH --container-image=/srv/home/users/loriss21cs/cfe/cfe.sqsh


#SBATCH --nodelist=dgx1
# # dgx-h100-em2
# # SBATCH --container-save=/srv/home/users/loriss21cs/cfe/cfe.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-7:59:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
# #SBATCH --partition=p_datamining
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=cba4442ed0417ba698aa1b18b17ee238f46945af

python3 --version

df -h
echo "http://localhost:2203/lab?token=gzr8ohuifnkjp90o3lewhuiopwelkd"
python3.10 -m jupyterlab --NotebookApp.token='gzr8ohuifnkjp90o3lewhuiopwelkd' --port=2203 --allow-root
