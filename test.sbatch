#!/bin/bash
#SBATCH --job-name="[CFE] Setup/Testing"

# #SBATCH --container-image=loris3/cfe:latest 
#SBATCH --container-image=/srv/home/users/loriss21cs/cfe/cfe.sqsh


# #SBATCH --nodelist=dgx-h100-em2,galadriel
#SBATCH --container-save=/srv/home/users/loriss21cs/cfe/cfe.sqsh
#SBATCH --container-mount-home 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/cfe
#SBATCH --nodes=1



python3 --version

df -h

# python3 trak_test.py --out ./
bash scripts/run_math_efficient.sh


echo $PYXIS_CONTAINER_IMAGE
echo $ENROOT_CACHE_PATH
echo $ENROOT_DATA_PATH 
echo $ENROOT_SQUASH_OPTIONS 
echo "end"