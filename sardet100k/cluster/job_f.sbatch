#!/bin/bash
# job_f: the GPU pair search (gpupairs.py, staged as ~/sbatch-staging/job_f.py with groups.py beside it) for every
# split pair, the planted-copy test and the shuffled-reference control. Images from the dataset tars plus
# sardet100k-planted.tar (recall.py's planted images), unpacked on the node's local disk by split.
# Output: runs/gp-<jobid>/gp_<query>_vs_<ref>.csv (and gp_val_vs_train_null.csv).
#   sbatch --job-name=job19 ~/sbatch-staging/job_f.sbatch
#SBATCH --job-name=job-f
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
L=${SLURM_TMPDIR:-/tmp}/gp-$SLURM_JOB_ID; OUT=runs/gp-$SLURM_JOB_ID
mkdir -p $L/img $OUT; trap 'rm -rf $L' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L images 2>&1 | grep -v "unknown extended header" || true
mv $L/images/train $L/images/val $L/img/
tar -xf ~/scratchDirectory/datasets/sardet100k-resplit.tar -C $L test_images 2>&1 | grep -v "unknown extended header" || true; mv $L/test_images $L/img/test
tar -xf ~/scratchDirectory/datasets/sardet100k-planted.tar -C $L/img 2>&1 | grep -v "unknown extended header" || true
find $L/img -name "._*" -delete  # macOS tar metadata files
echo "unpacked $(date '+%H:%M'): $(ls $L/img/train | wc -l) train, $(ls $L/img/val | wc -l) val, $(ls $L/img/test | wc -l) test, $(ls $L/img/planted | wc -l) planted"
run() { echo "== $* $(date '+%H:%M')"; python -u ~/sbatch-staging/job_f.py $L/img "$@"; }
run planted train $OUT/gp_planted_vs_train.csv
run val train $OUT/gp_val_vs_train_null.csv --null
run val train $OUT/gp_val_vs_train.csv
run test train $OUT/gp_test_vs_train.csv
run val test $OUT/gp_val_vs_test.csv
run val val $OUT/gp_val_vs_val.csv
run test test $OUT/gp_test_vs_test.csv
run train train $OUT/gp_train_vs_train.csv
ls -la $OUT
echo "done $(date '+%m-%d %H:%M')"
