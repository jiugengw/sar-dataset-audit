#!/bin/bash
# job_ad: RSAR score inflation. A rotated-box YOLO11s (yolo11s-obb, COCO-pretrained yolo11s backbone, the SARDet-100K
# recipe: 24 epochs, 512 px, batch 32) trained on RSAR's official split (FOLD=official; scored on val and test) or on
# one fold of the SARDet-100K group cross-validation (FOLD=0..4; scored on the held-out fold), overall and per source.
#   FOLD=official sbatch --job-name=job59 ~/sbatch-staging/job_ad.sbatch
# Needs ~/sbatch-staging/rsar/{prepare.py,score_obb.py,cv_folds.csv}. Output: runs/rsar-obb-<FOLD>-<jobid>/scores.json.
#SBATCH --job-name=job-ad
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
export MPLCONFIGDIR=/tmp/job-ad-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}; FOLD=${FOLD:?set FOLD=official or 0..4}; R=~/sbatch-staging/rsar
L=${SLURM_TMPDIR:-/tmp}/rsar-obb-$SLURM_JOB_ID; OUT=runs/rsar-obb-$FOLD-$SLURM_JOB_ID
mkdir -p $L/raw $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); fold $FOLD"
echo "cv_folds md5 $(md5sum < $R/cv_folds.csv)"
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 15000000 ] || { echo "need 15 GB free on $L"; exit 3; }
tar -xf ~/scratchDirectory/datasets/RSAR.tar -C $L/raw
python -u $R/prepare.py $L/raw $R/cv_folds.csv $FOLD $L/ds | tee $OUT/prepare.json
echo "laid out $(date '+%H:%M'): $(ls $L/ds/images/train | wc -l) train, $(ls $L/ds/images/test | wc -l) test; left over $(find $L/raw -name '*.*' -path '*images*' | wc -l)"
[ "$(ls $L/ds/images/test | wc -l)" -gt 5000 ] || { echo "test part is (almost) empty: stop"; exit 4; }
yolo obb train data=$L/ds/data.yaml model=yolo11s-obb.yaml pretrained=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS \
    batch=32 workers=4 val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True
python -u $R/score_obb.py $OUT/train/weights/last.pt $L/ds/eval $OUT/scores.json
ls -la $OUT/train/weights $OUT/scores.json
echo "done $(date '+%m-%d %H:%M')"
