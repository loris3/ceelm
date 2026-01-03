#!/bin/bash
#SBATCH --job-name="[CEELM] Jupyter Lab Server"
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:0
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
#SBATCH --container-writable


source .env
ulimit -n 65535

pip install ipykernel==6.30.1

python3 --version

df -h
echo "http://localhost:2203/lab?token=$JUPYTER_TOKEN"
python3.10 -m jupyterlab --NotebookApp.token="$JUPYTER_TOKEN" --port=2203 --allow-root
