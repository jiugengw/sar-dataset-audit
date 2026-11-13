#!/bin/bash
# job_c: the same YOLO11s run as job_a, but on the leak-free re-split (resplit.py): train on the new train, predict
# the new val and the new test. Comparing its val mAP with job_a's (official split, same recipe, same sizes and
# source mix) measures how much the leaks inflate the official val score.
# Needs ~/scratchDirectory/datasets/sardet100k-yolo.tar (official train/val images) and sardet100k-resplit.tar
# (test images, labels of the new split, split.txt; made by yolo_resplit.py). Both are unpacked on the node's disk.
#   sbatch --job-name=job17 ~/sbatch-staging/job_c.sbatch
#SBATCH --job-name=job-c
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
export MPLCONFIGDIR=/tmp/job-c-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}
L=${SLURM_TMPDIR:-/tmp}/rs-$SLURM_JOB_ID; OUT=runs/rs-$SLURM_JOB_ID
mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); local disk:"
df -h $L | tail -1
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 20000000 ] || { echo "need 20 GB free on $L"; exit 3; }
mkdir -p $L/off $L/images/train $L/images/val $L/images/test
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L/off images 2>&1 | grep -v "unknown extended header" || true
tar -xf ~/scratchDirectory/datasets/sardet100k-resplit.tar -C $L
mv $L/test_images $L/off/images/test
# move every image into its new split (file names are unique across the official splits)
while read -r off f new; do mv "$L/off/images/$off/$f" "$L/images/$new/$f"; done < $L/split.txt
{ printf 'path: %s\ntrain: images/train\nval: images/val\nnames:\n' $L; awk '{printf "  %d: %s\n", NR-1, $1}' $L/names.txt; } > $L/data.yaml
cat $L/data.yaml
echo "unpacked $(date '+%H:%M'): $(ls $L/images/train | wc -l) train, $(ls $L/images/val | wc -l) val, $(ls $L/images/test | wc -l) test;" \
     "left over: $(find $L/off/images -type f | wc -l)"
yolo detect train data=$L/data.yaml model=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS batch=32 workers=4 \
    val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/val $OUT/val_predictions.json
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/test $OUT/test_predictions.json
ls -la $OUT/train/weights $OUT/*_predictions.json
echo "done $(date '+%m-%d %H:%M')"
