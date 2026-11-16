#!/bin/bash
# Queue the RSAR rotated-box runs (job_ad: official split, then folds 0..4), one by one as the 4-submitted-jobs limit
# allows; then, when they have all left the queue, write the summary (rsar/summary.py) on the login node. Log: ~/sbatch-staging/queue_after.log
#   NAME_BASE=59 nohup ~/sbatch-staging/queue_after.sh >> ~/sbatch-staging/queue_after.log 2>&1 < /dev/null &
cd ~/scratchDirectory/sar-benchmark
B=${NAME_BASE:?}; NAMES=""
i=0
for f in official 0 1 2 3 4; do
  while [ "$(squeue -h -u $USER | wc -l)" -ge 4 ]; do sleep 300; done
  n=job$((B + i)); i=$((i + 1)); NAMES=$NAMES${NAMES:+,}$n
  FOLD=$f sbatch --job-name=$n ~/sbatch-staging/job_ad.sbatch
  echo "$(date '+%m-%d %H:%M') submitted $f as $n"
done
while [ -n "$(squeue -h -u $USER -n $NAMES)" ]; do sleep 600; done
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
mkdir -p runs/rsar-obb-summary; python ~/sbatch-staging/rsar/summary.py runs runs/rsar-obb-summary/summary.json
echo "$(date '+%m-%d %H:%M') all done"
