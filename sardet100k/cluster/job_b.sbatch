#!/bin/bash
# job_b: predict SARDet-100K val with already-trained weights (job_a's prediction step on its own).
# Unpacks only images/val (10,492 files) from the dataset tar onto the node's local disk.
#   W=runs/detect/runs/sd-364677/train/weights/last.pt OUT=runs/sd-364677 sbatch --job-name=job16 ~/sbatch-staging/job_b.sbatch
#SBATCH --job-name=job-b
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export MPLCONFIGDIR=/tmp/job-b-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
L=${SLURM_TMPDIR:-/tmp}/sdb-$SLURM_JOB_ID; mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L images/val 2>&1 | grep -v "unknown extended header" || true
echo "unpacked $(date '+%H:%M'): $(ls $L/images/val | wc -l) val"
python -u ~/sbatch-staging/job_a.py $W $L/images/val $OUT/val_predictions.json
ls -la $OUT/val_predictions.json
echo "done $(date '+%m-%d %H:%M')"
