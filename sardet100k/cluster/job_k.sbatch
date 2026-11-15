#!/bin/bash
# job_k: why the version-2 control still confirms links (gp2_null.py): v2 (job23, runs/gq-365537) next to v1 (job19, runs/gp-365121).
#   sbatch --job-name=job28 ~/sbatch-staging/job_k.sbatch
#SBATCH --job-name=job-k
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
python -u $A/gp2_null.py $R runs/gp-365121 $A/results/resplit_split.csv $R/gp2_null.json
echo "done $(date '+%m-%d %H:%M')"
