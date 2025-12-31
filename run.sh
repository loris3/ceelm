#!/bin/bash
jobid=$(sbatch selection_AIDE.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jobid validation_AIDE.sbatch
sbatch --dependency=afterok:$jobid score_AIDE.sbatch



jobid=$(sbatch selection_DIVINE.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jobid validation_DIVINE.sbatch
sbatch --dependency=afterok:$jobid score_DIVINE.sbatch


jobid=$(sbatch selection_FL.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jobid validation_FL.sbatch
sbatch --dependency=afterok:$jobid score_FL.sbatch

sbatch score_random.sbatch
sbatch score_naive.sbatch
sbatch score_self.sbatch

sbatch validation_naive.sbatch
sbatch validation_FL.sbatch
