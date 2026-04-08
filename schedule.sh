#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <job-file.sbatch>"
  exit 1
fi

JOB_FILE="$1"




source .env
SBATCH_ARGS=()
if [[ "$1" == --dependency=* ]]; then
    SBATCH_ARGS+=("$1")
    shift
fi
SBATCH_ARGS+=(--container-image="$CONTAINER_IMAGE")
SBATCH_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")
SBATCH_ARGS+=(--nodelist="$NODELIST")
SBATCH_ARGS+=(--container-mount-home)

sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE"
