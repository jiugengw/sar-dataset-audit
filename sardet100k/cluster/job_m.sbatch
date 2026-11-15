#!/bin/bash
# job_m: which links join different sources into the largest group of the rebuilt re-split (bridges.py).
# runs/gq-365537, incl. gp_msar_numbers.csv), name links, contained chips and the earlier CPU searches, under the
# groups.py rules. Staged in ~/sbatch-staging/audit/ (code, results/ inputs, data/SARDet_100K annotations and
# mapping.json). Output: runs/br-<jobid>/resplit.json, resplit_split.csv (and the split it replaces).
#   sbatch --job-name=job30 ~/sbatch-staging/job_m.sbatch
#SBATCH --job-name=job-m
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
A=~/sbatch-staging/audit; G=runs/gq-365537; OUT=runs/br-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
python -u $A/bridges.py $OUT/bridges.json
echo "done $(date '+%m-%d %H:%M')"
