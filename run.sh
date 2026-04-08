#!/bin/bash
source .env

SBATCH_ARGS=()
SBATCH_ARGS+=(--container-image="$CONTAINER_IMAGE")
SBATCH_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")


SBATCH_ARGS+=(--nodelist="$NODELIST")

SBATCH_ARGS+=(--container-mount-home)







jobid=$(sbatch "${SBATCH_ARGS[@]}" selection_AIDE.sbatch | awk '{print $4}')
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid score_AIDE.sbatch
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid validation_AIDE.sbatch




jobid=$(sbatch "${SBATCH_ARGS[@]}" selection_DIVINE.sbatch | awk '{print $4}')
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid validation_DIVINE.sbatch
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid score_DIVINE.sbatch


jobid=$(sbatch "${SBATCH_ARGS[@]}" selection_FL.sbatch | awk '{print $4}')
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid score_FL.sbatch
sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:$jobid validation_FL.sbatch


sbatch "${SBATCH_ARGS[@]}" score_random.sbatch
sbatch "${SBATCH_ARGS[@]}" validation_random.sbatch
sbatch "${SBATCH_ARGS[@]}" score_naive.sbatch
sbatch "${SBATCH_ARGS[@]}" validation_naive.sbatch
sbatch "${SBATCH_ARGS[@]}" score_self.sbatch
sbatch "${SBATCH_ARGS[@]}" validation_self.sbatch


