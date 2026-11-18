#!/bin/bash
# job_ag: M4-SAR score inflation. A rotated-box YOLO11s (yolo11s-obb, COCO-pretrained yolo11s backbone, the recipe used
# for SARDet-100K and RSAR: 24 epochs, 512 px, batch 32) on M4-SAR's SAR images (both versions), trained on the official
# split (FOLD=official; scored on val and test) or on one leak-free fold (FOLD=0..4 with FOLDS=<csv id,fold>).
#   FOLD=official sbatch --job-name=job72 ~/sbatch-staging/job_ag.sbatch
# Needs ~/sbatch-staging/m4sar/prepare.py and ~/sbatch-staging/rsar/score_obb.py. Output: runs/m4-obb-<FOLD>-<jobid>/scores.json.
#SBATCH --job-name=job-ag
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
export MPLCONFIGDIR=/tmp/job-ag-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}; FOLD=${FOLD:?set FOLD=official or 0..4}; S=~/sbatch-staging
L=${SLURM_TMPDIR:-/tmp}/m4-obb-$SLURM_JOB_ID; OUT=runs/m4-obb-$FOLD-$SLURM_JOB_ID
mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); fold $FOLD"
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 25000000 ] || { echo "need 25 GB free on $L"; exit 3; }
[ "$FOLD" = official ] || echo "folds md5 $(md5sum < ${FOLDS:?set FOLDS for a fold run})"
python -u $S/m4sar/prepare.py ~/scratchDirectory/datasets/M4-SAR.zip $FOLD $L/ds ${FOLDS:-} | tee $OUT/prepare.json
echo "laid out $(date '+%H:%M'): $(ls $L/ds/images/train | wc -l) train, $(ls $L/ds/images/test | wc -l) test"
yolo obb train data=$L/ds/data.yaml model=yolo11s-obb.yaml pretrained=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS \
    batch=32 workers=4 val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True
python -u $S/rsar/score_obb.py $OUT/train/weights/last.pt $L/ds/eval $OUT/scores.json
ls -la $OUT/train/weights $OUT/scores.json
echo "done $(date '+%m-%d %H:%M')"
