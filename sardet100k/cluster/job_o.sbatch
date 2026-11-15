#!/bin/bash
# job_o: one fold of the 5-fold group cross-validation (folds.py): YOLO11s, same recipe as job_a / job_c, trained on
# the other four folds and scored on fold FOLD (score_fold.py). Folds are made on the node from the final groups
# (~/sbatch-staging/audit/results/resplit_split.csv); every run checks it got the same folds (md5 in the log).
#   FOLD=0 sbatch --job-name=job38 ~/sbatch-staging/job_o.sbatch
#SBATCH --job-name=job-o
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
export MPLCONFIGDIR=/tmp/job-o-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}; FOLD=${FOLD:?set FOLD=0..4}; A=~/sbatch-staging/audit
L=${SLURM_TMPDIR:-/tmp}/cv-$SLURM_JOB_ID; OUT=runs/cv-fold$FOLD-$SLURM_JOB_ID
mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); fold $FOLD"
python -u $A/folds.py $A/results/resplit_split.csv $A/data/SARDet_100K $OUT/folds.csv | tee $OUT/folds.json
echo "folds md5 $(md5sum < $OUT/folds.csv)"
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 20000000 ] || { echo "need 20 GB free on $L"; exit 3; }
mkdir -p $L/off $L/images/train $L/images/test $L/labels/train $L/labels/test
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L/off images 2>&1 | grep -v "unknown extended header" || true
tar -xf ~/scratchDirectory/datasets/sardet100k-resplit.tar -C $L/off test_images labels names.txt split.txt
mv $L/off/test_images $L/off/images/test
# labels sit in the tar under the provisional split (split.txt); move images and labels by fold
declare -A OLD; while read -r off f new; do OLD[$f]=$new; done < $L/off/split.txt
tail -n +2 $OUT/folds.csv | while IFS=, read -r off f src grp fd; do
  fd=${fd%$'\r'}
  [ "$fd" = "$FOLD" ] && to=test || to=train
  mv "$L/off/images/$off/$f" "$L/images/$to/$f"; s=${f%.*}; mv "$L/off/labels/${OLD[$f]}/$s.txt" "$L/labels/$to/$s.txt"
done
{ printf 'path: %s\ntrain: images/train\nval: images/test\nnames:\n' $L; awk '{printf "  %d: %s\n", NR-1, $1}' $L/off/names.txt; } > $L/data.yaml
echo "unpacked $(date '+%H:%M'): $(ls $L/images/train | wc -l) train, $(ls $L/images/test | wc -l) test;" \
     "labels $(ls $L/labels/train | wc -l) / $(ls $L/labels/test | wc -l); left over: $(find $L/off/images $L/off/labels -type f | wc -l)"
[ "$(ls $L/images/test | wc -l)" -gt 10000 ] || { echo "test fold is (almost) empty: stop"; exit 4; }
yolo detect train data=$L/data.yaml model=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS batch=32 workers=4 \
    val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/test $OUT/test_predictions.json
python -u $A/score_fold.py $OUT/folds.csv $FOLD $OUT/test_predictions.json $OUT/scores_fold.json
ls -la $OUT/train/weights $OUT/test_predictions.json
echo "done $(date '+%m-%d %H:%M')"
