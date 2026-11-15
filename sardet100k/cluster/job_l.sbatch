#!/bin/bash
# job_l: rebuild the leak-free re-split (resplit.py) with every link: the version-2 GPU search (job23,
# runs/gq-365537, incl. gp_msar_numbers.csv), name links, contained chips and the earlier CPU searches, under the
# groups.py rules. Staged in ~/sbatch-staging/audit/ (code, results/ inputs, data/SARDet_100K annotations and
# mapping.json). Output: runs/rb-<jobid>/resplit.json, resplit_split.csv (and the split it replaces).
#   sbatch --job-name=job29 ~/sbatch-staging/job_l.sbatch
#SBATCH --job-name=job-l
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
A=~/sbatch-staging/audit; G=runs/gq-365537; OUT=runs/rb-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
cp $A/results/resplit_split.csv $OUT/resplit_split_before.csv
for f in gp_val_vs_train gp_test_vs_train gp_val_vs_test gp_val_vs_val gp_test_vs_test gp_msar_numbers; do
  cp $G/$f.csv $A/results/$f.csv; done
cp $G/gp_train_vs_train.csv.gz $A/results/
ls -la $A/results | grep gp_
python -u $A/resplit.py | tail -n 60
cp $A/results/resplit.json $A/results/resplit_split.csv $OUT/
echo "done $(date '+%m-%d %H:%M')"
