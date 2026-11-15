#!/bin/bash
# job_n: share of official val / test images leaked into train under the final rules (leakpct.py).
# runs/gq-365537, incl. gp_msar_numbers.csv), name links, contained chips and the earlier CPU searches, under the
# groups.py rules. Staged in ~/sbatch-staging/audit/ (code, results/ inputs, data/SARDet_100K annotations and
# mapping.json). Output: runs/lp-<jobid>/resplit.json, resplit_split.csv (and the split it replaces).
#   sbatch --job-name=job35 ~/sbatch-staging/job_n.sbatch
#SBATCH --job-name=job-n
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
A=~/sbatch-staging/audit; G=runs/gq-365537; OUT=runs/lp-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
python -u $A/leakpct.py $OUT/leakpct.json
echo "done $(date '+%m-%d %H:%M')"
