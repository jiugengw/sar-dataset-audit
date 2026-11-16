#!/bin/bash
# job_p: combine the 5 cross-validation folds (job_o). Takes the newest finished run of each fold
# (runs/cv-fold<F>-<jobid> with scores_fold.json), checks every fold used the same folds.csv (md5), then scores all
# images together, each with the model that never saw it (score_fold.py ... pooled), and writes per-fold, mean and
# spread summaries. Output: runs/cv-pooled-<jobid>/{scores_pooled.json,summary.json}.
#   sbatch --job-name=job48 ~/sbatch-staging/job_p.sbatch
#SBATCH --job-name=job-p
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
A=~/sbatch-staging/audit; OUT=runs/cv-pooled-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
RUNS=(); PREDS=()
for f in 0 1 2 3 4; do
  d=$(ls -dt runs/cv-fold$f-* | while read -r x; do [ -f $x/scores_fold.json ] && { echo $x; break; }; done)
  [ -n "$d" ] || { echo "fold $f has no finished run: stop"; exit 2; }
  echo "fold $f: $d  folds md5 $(md5sum < $d/folds.csv)"
  RUNS+=($d); PREDS+=($d/test_predictions.json)
done
[ "$(for d in ${RUNS[@]}; do md5sum < $d/folds.csv; done | sort -u | wc -l)" = 1 ] || { echo "folds.csv differ: stop"; exit 3; }
cp ${RUNS[0]}/folds.csv $OUT/folds.csv
python -u $A/score_fold.py $OUT/folds.csv pooled $(IFS=,; echo "${PREDS[*]}") $OUT/scores_pooled.json
python -u - $OUT ${RUNS[@]} <<'EOF'
import json, statistics as st, sys
from pathlib import Path
out, runs = Path(sys.argv[1]), sys.argv[2:]
folds = [json.load(open(f'{r}/scores_fold.json')) for r in runs]
pooled = json.load(open(out / 'scores_pooled.json'))
keys = list(folds[0]['all + sources'])
summ = {'runs': runs, 'per fold mAP': {}, 'mean / sd / min / max (mAP)': {}, 'pooled mAP': {}, 'groups': {}}
for k in keys:
    v = [f['all + sources'][k]['mAP'] for f in folds if k in f['all + sources']]
    summ['per fold mAP'][k] = v
    summ['mean / sd / min / max (mAP)'][k] = [round(st.mean(v), 2), round(st.stdev(v), 2), min(v), max(v)]
    summ['pooled mAP'][k] = pooled['all + sources'][k]['mAP']
for i, f in enumerate(folds):
    for g, v in f['groups (100+ images)'].items():
        summ['groups'][g] = {'fold': i, **v}
(out / 'summary.json').write_text(json.dumps(summ, indent=1))
for k in keys:
    m = summ['mean / sd / min / max (mAP)'][k]
    print(f"{k:24s} folds {summ['per fold mAP'][k]}  mean {m[0]} sd {m[1]}  pooled {summ['pooled mAP'][k]}")
EOF
echo "done $(date '+%m-%d %H:%M')"
