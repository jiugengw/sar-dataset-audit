#!/bin/bash
# job_a: train YOLO11s on SARDet-100K train (official split) and predict val, for the leaked-vs-clean scoring.
# The dataset is one tar on scratch (~/scratchDirectory/datasets/sardet100k-yolo.tar, made by yolo_prepare.py)
# and is unpacked on the node's local disk: 210k image + label files would not fit the 100k-file scratch quota.
# Only the weights and the val predictions are kept, in runs/sd-<jobid>/.
#   sbatch --job-name=job15 ~/sbatch-staging/job_a.sbatch
#SBATCH --job-name=job-a
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
export MPLCONFIGDIR=/tmp/job-a-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}
L=${SLURM_TMPDIR:-/tmp}/sd-$SLURM_JOB_ID; OUT=runs/sd-$SLURM_JOB_ID
mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); local disk:"
df -h $L | tail -1
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 20000000 ] || { echo "need 20 GB free on $L"; exit 3; }
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L
rm -f $L/labels/*.cache
printf 'path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ship\n  1: aircraft\n  2: car\n  3: tank\n  4: bridge\n  5: harbor\n' $L > $L/data.yaml
echo "unpacked $(date '+%H:%M'): $(ls $L/images/train | wc -l) train, $(ls $L/images/val | wc -l) val"
yolo detect train data=$L/data.yaml model=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS batch=32 workers=4 \
    val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True  # absolute: a relative project lands under runs/detect/
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/val $OUT/val_predictions.json
ls -la $OUT/train/weights $OUT/val_predictions.json
echo "done $(date '+%m-%d %H:%M')"
