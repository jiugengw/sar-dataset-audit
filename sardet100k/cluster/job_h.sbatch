#!/bin/bash
# job_h: job_c (YOLO11s on the leak-free re-split) on a redrawn split: the same groups, dealt to val / test with another
# seed (redraw.py), so other airports and scenes land in val and test. Comparing it with job_c shows how much the
# score moves with the draw alone. Redraws, trains, predicts and scores on the node.
# Staged in ~/sbatch-staging/audit/: assign.py redraw.py resplit.py groups.py score_resplit.py score_draw.py,
# results/resplit_split.csv, data/SARDet_100K/{Annotations/train.json,Annotations/val.json,test.json}.
#   SEED=1 sbatch --job-name=job24 ~/sbatch-staging/job_h.sbatch
#SBATCH --job-name=job-h
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
export MPLCONFIGDIR=/tmp/job-h-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1
EPOCHS=${EPOCHS:-24}; SEED=${SEED:-1}; A=~/sbatch-staging/audit
L=${SLURM_TMPDIR:-/tmp}/rd-$SLURM_JOB_ID; OUT=runs/rd-$SLURM_JOB_ID
mkdir -p $L $OUT; trap 'rm -rf $L $MPLCONFIGDIR' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader); seed $SEED"
# seed 0 must give back the split job_c trained on, else the redraw code differs from resplit.py
python -u $A/redraw.py $A/results/resplit_split.csv 0 $OUT/split_seed0.txt $A/data/SARDet_100K | grep same
python - $A/results/resplit_split.csv $OUT/split_seed0.txt <<'PY'
import csv, sys
saved = {r['file_name']: r['new_split'] for r in csv.DictReader(open(sys.argv[1]))}
redo = dict(l.split()[1:] for l in open(sys.argv[2]))
bad = sum(saved[f] != s for f, s in redo.items()); print('seed 0 differs from the saved split on', bad, 'images')
sys.exit(1 if bad else 0)
PY
python -u $A/redraw.py $A/results/resplit_split.csv $SEED $OUT/split.txt $A/data/SARDet_100K | tee $OUT/redraw.json
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 20000000 ] || { echo "need 20 GB free on $L"; exit 3; }
mkdir -p $L/off $L/images/train $L/images/val $L/images/test $L/labels/train $L/labels/val $L/labels/test
tar -xf ~/scratchDirectory/datasets/sardet100k-yolo.tar -C $L/off images 2>&1 | grep -v "unknown extended header" || true
tar -xf ~/scratchDirectory/datasets/sardet100k-resplit.tar -C $L/off test_images labels names.txt split.txt
mv $L/off/test_images $L/off/images/test
# images by the redrawn split; labels sit in the tar under the seed-0 split, so move them too
declare -A OLD; while read -r off f new; do OLD[$f]=$new; done < $L/off/split.txt
while read -r off f new; do
  mv "$L/off/images/$off/$f" "$L/images/$new/$f"; s=${f%.*}; mv "$L/off/labels/${OLD[$f]}/$s.txt" "$L/labels/$new/$s.txt"
done < $OUT/split.txt
{ printf 'path: %s\ntrain: images/train\nval: images/val\nnames:\n' $L; awk '{printf "  %d: %s\n", NR-1, $1}' $L/off/names.txt; } > $L/data.yaml
echo "unpacked $(date '+%H:%M'): $(ls $L/images/train | wc -l) train, $(ls $L/images/val | wc -l) val, $(ls $L/images/test | wc -l) test;" \
     "labels $(ls $L/labels/train | wc -l) / $(ls $L/labels/val | wc -l) / $(ls $L/labels/test | wc -l); left over: $(find $L/off/images $L/off/labels -type f | wc -l)"
yolo detect train data=$L/data.yaml model=pretrained/yolo11s.pt imgsz=512 epochs=$EPOCHS batch=32 workers=4 \
    val=False plots=False device=0 project=$PWD/$OUT name=train exist_ok=True
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/val $OUT/val_predictions.json
python -u ~/sbatch-staging/job_a.py $OUT/train/weights/last.pt $L/images/test $OUT/test_predictions.json
python -u $A/score_draw.py $OUT/split.txt $OUT/val_predictions.json $OUT/test_predictions.json $OUT/scores_draw.json
ls -la $OUT/train/weights $OUT/*_predictions.json
echo "done $(date '+%m-%d %H:%M')"
