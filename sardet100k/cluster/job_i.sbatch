#!/bin/bash
# job_i: per-airport scores (per_group.py) for the two re-split draws: job22 (seed 0, runs/rs-365510) and job24
# (seed 1, runs/rd-365587). CPU work only; staged in ~/sbatch-staging/audit/ like job_h.
#   sbatch --job-name=job25 ~/sbatch-staging/job_i.sbatch
#SBATCH --job-name=job-i
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
A=~/sbatch-staging/audit; R1=runs/rs-365510; R2=runs/rd-365587; OUT=runs/pg-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
python -u $A/per_group.py $OUT/per_group.json \
    "draw1:saved:$R1/val_predictions.json:$R1/test_predictions.json" \
    "draw2:$R2/split.txt:$R2/val_predictions.json:$R2/test_predictions.json"
echo "done $(date '+%m-%d %H:%M')"
