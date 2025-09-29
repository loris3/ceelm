#!/bin/bash
#SBATCH --job-name="[CFE] Merging Parquet Files"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1,shelob
#SBATCH --container-mount-home 
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu:0
#SBATCH --time=0-00:59:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1

python3 --version

df -h
ulimit -n 65535

echo "Start"
python3 merge_results.py
echo "Done!"