#!/bin/bash
# job_af: M4-SAR copy search on the pixels with its null and planted-copy controls (m4sar/pixsearch.py), read straight
# from ~/scratchDirectory/datasets/M4-SAR.zip. Fits a 24 GB GPU (peak ~20 GB: the reference patches take 18 GB). LIMIT=2000 for a trial.
# Resume after a stop: CANDS=runs/m4-search-<id>/cands_main.npz MAIN_DONE=runs/m4-search-<id>/pairs_main.csv sbatch ...
#   sbatch --job-name=job66 ~/sbatch-staging/job_af.sbatch
# Output: runs/m4-search-<jobid>/.
#SBATCH --job-name=job-af
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
OUT=runs/m4-search-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); limit ${LIMIT:-none}"
python -m py_compile ~/sbatch-staging/m4sar/pixsearch.py
python -u ~/sbatch-staging/m4sar/pixsearch.py ~/scratchDirectory/datasets/M4-SAR.zip $OUT
echo "done $(date '+%m-%d %H:%M')"
