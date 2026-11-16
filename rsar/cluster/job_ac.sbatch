#!/bin/bash
# job_ac: RSAR's leak rate from the SARDet-100K links (rsar/leak.py), given the RSAR -> SARDet match (job_ab).
#   MATCH=runs/rsar-match-<jobid>/rsar_to_sardet.csv sbatch --job-name=job57 --dependency=afterok:<jobid> ~/sbatch-staging/job_ac.sbatch
# Output: runs/rsar-leak-<jobid>/leak.json.
#SBATCH --job-name=job-ac
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
MATCH=${MATCH:?set MATCH=runs/rsar-match-<jobid>/rsar_to_sardet.csv}; OUT=runs/rsar-leak-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname); match $MATCH"
python -u ~/sbatch-staging/rsar/leak.py $MATCH ~/sbatch-staging/audit $OUT/leak.json
echo "done $(date '+%m-%d %H:%M')"
