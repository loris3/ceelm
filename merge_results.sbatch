#!/bin/bash
#SBATCH --job-name="[CEELM] Merging Parquet Files"
#SBATCH --container-image="ghcr.io#loris3/cfe:latest"
#SBATCH --nodelist=dgx-h100-em2,dgx1,shelob,galadriel
#SBATCH --container-mount-home 
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64  
#SBATCH --time=0-01:59:00
#SBATCH --container-workdir=/mnt/nlp-data/home/users/loriss21cs/cfe
#SBATCH --nodes=1
python3 --version

df -h
ulimit -n 65535

echo "Start"
python3 merge_results.py --source_dir="cache/validation/partial" --target_dir="results/validation"
python3 merge_results.py --source_dir="cache/scoring/partial" --target_dir="results/scoring"
echo "Done!"
