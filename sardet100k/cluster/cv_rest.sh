#!/bin/bash
# Submit cross-validation folds (FOLDS, default 2 3 4) (job_o) as queue slots free up (4 submitted jobs per user). Log: ~/sbatch-staging/cv_rest.log
cd ~/scratchDirectory/sar-benchmark
for f in ${FOLDS:-2 3 4}; do
  while [ "$(squeue -h -u $USER | wc -l)" -ge 4 ]; do sleep 120; done
  FOLD=$f sbatch --job-name=job$((38 + f)) ~/sbatch-staging/job_o.sbatch
  echo "$(date '+%m-%d %H:%M') submitted fold $f"
done
echo "$(date '+%m-%d %H:%M') all folds submitted"
