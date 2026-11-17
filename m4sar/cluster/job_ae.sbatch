#!/bin/bash
# job_ae: M4-SAR identical-label twins across splits, checked on the pixels (m4sar/labeltwins.py), read straight
# from ~/scratchDirectory/datasets/M4-SAR.zip (huggingface.co/datasets/wchao0601/m4-sar, sha256 7ca1ae75...).
#   sbatch --job-name=job65 ~/sbatch-staging/job_ae.sbatch
# Output: runs/m4-twins-<jobid>/.
#SBATCH --job-name=job-ae
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OUT=runs/m4-twins-$SLURM_JOB_ID; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
python -u ~/sbatch-staging/m4sar/labeltwins.py ~/scratchDirectory/datasets/M4-SAR.zip $OUT
echo "done $(date '+%m-%d %H:%M')"
