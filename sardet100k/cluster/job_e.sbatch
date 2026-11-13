#!/bin/bash
# job_e: small images inside larger ones (contained.py, staged as ~/sbatch-staging/job_e.py) on the GPU, then the
# same search against block-shuffled hosts for 10,000 chips (the null). Images come from the two dataset tars,
# unpacked on the node's local disk by official split; resplit_split.csv (staged as job_e_split.csv) lists them.
# Output: runs/ct-<jobid>/contained.csv and contained_null.csv.
#   sbatch --job-name=job18 ~/sbatch-staging/job_e.sbatch
#SBATCH --job-name=job-e
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
L=${SLURM_TMPDIR:-/tmp}/ct-$SLURM_JOB_ID; OUT=runs/ct-$SLURM_JOB_ID
mkdir -p $L/img $OUT; trap 'rm -rf $L' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L images 2>&1 | grep -v "unknown extended header" || true
mv $L/images/train $L/images/val $L/img/
tar -xf ~/scratchDirectory/datasets/sardet100k-resplit.tar -C $L test_images
mv $L/test_images $L/img/test
echo "unpacked $(date '+%H:%M'): $(ls $L/img/train | wc -l) train, $(ls $L/img/val | wc -l) val, $(ls $L/img/test | wc -l) test"
python -u ~/sbatch-staging/job_e.py $L/img ~/sbatch-staging/job_e_split.csv $OUT/contained.csv
python -u ~/sbatch-staging/job_e.py $L/img ~/sbatch-staging/job_e_split.csv $OUT/contained_null.csv --null --chips 10000
ls -la $OUT
echo "done $(date '+%m-%d %H:%M')"
