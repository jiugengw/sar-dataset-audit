#!/bin/bash
# job_j: CPU checks of the version-2 GPU search (job23, runs/gq-365537): planted recall and null false links per
# confirmation rule (gp2_eval.py). Staged in ~/sbatch-staging/audit/ with groups.py and results/recall_truth.csv.
#   sbatch --job-name=job26 ~/sbatch-staging/job_j.sbatch
#SBATCH --job-name=job-j
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
A=~/sbatch-staging/audit; R=runs/gq-365537
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
python -u $A/gp2_eval.py $R $A/results/recall_truth.csv $R/gp2_eval.json
echo "done $(date '+%m-%d %H:%M')"
