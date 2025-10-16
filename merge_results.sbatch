#!/bin/bash
#SBATCH --job-name="[CFE] Merging Parquet Files"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1,shelob,galadriel
#SBATCH --container-mount-home 
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu:0
#SBATCH --time=0-00:59:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1

python3 --version

df -h
ulimit -n 65535

echo "Start"
python3 merge_results.py --source_dir="cache/validation/partial" --target_dir="results/validation"
python3 merge_results.py --source_dir="cache/scoring/partial" --target_dir="results/scoring"
echo "Done!"