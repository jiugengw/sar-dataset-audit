#!/bin/bash
# job_ah: M4-SAR scale check (m4sar/scalesearch.py): copies at zoom 1.1-2x between val / test and train, with its null
# and planted-copy controls, read straight from ~/scratchDirectory/datasets/M4-SAR.zip. Needs pixsearch.py next to it.
# Trial: LIMIT=2000 N_NULL=300. Resume after a stop: RESUME=runs/m4-scale-<id> (reuses its cands_*.npz).
#   sbatch --job-name=job79 --constraint="l40s|a40" ~/sbatch-staging/job_ah.sbatch
# Output: runs/m4-scale-<jobid>/.
#SBATCH --job-name=job-ah
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/m4-scale-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); limit ${LIMIT:-none}"
python -m py_compile ~/sbatch-staging/m4sar/scalesearch.py
python -u ~/sbatch-staging/m4sar/scalesearch.py ~/scratchDirectory/datasets/M4-SAR.zip $OUT
echo "done $(date '+%m-%d %H:%M')"
