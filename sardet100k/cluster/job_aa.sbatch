#!/bin/bash
# job_aa: the official-split YOLO11s (job_a / job_b predictions, staged as audit/results/{val,test}_predictions.json)
# on leaked vs clean official val / test images, where 'leaked' = a direct confirmed link to train under the final
# rules (score_leak.py; the same set as leakpct.py). Output: runs/sl-<jobid>/scores_leak.json.
#   sbatch --job-name=job49 ~/sbatch-staging/job_aa.sbatch
#SBATCH --job-name=job-aa
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
A=~/sbatch-staging/audit; OUT=runs/sl-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
md5sum $A/results/val_predictions.json $A/results/test_predictions.json
python -u $A/score_leak.py $OUT/scores_leak.json
echo "done $(date '+%m-%d %H:%M')"
