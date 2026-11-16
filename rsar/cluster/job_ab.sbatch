#!/bin/bash
# job_ab: match every RSAR image to its SARDet-100K image by pixels (rsar/match.py). Unpacks RSAR.tar
# (~/scratchDirectory/datasets/RSAR.tar, from the authors' Google Drive link) and the SARDet-100K official images
# (sardet100k-yolo.tar train / val, sardet100k-resplit.tar test) on the node's local disk.
# Output: runs/rsar-match-<jobid>/{rsar_to_sardet.csv,match.json}.
#   sbatch --job-name=job56 ~/sbatch-staging/job_ab.sbatch
#SBATCH --job-name=job-ab
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
D=~/scratchDirectory/datasets; L=${SLURM_TMPDIR:-/tmp}/rsar-$SLURM_JOB_ID; OUT=runs/rsar-match-$SLURM_JOB_ID
mkdir -p $L/rsar $L/sd $OUT; trap 'rm -rf $L' EXIT
echo "start $(date '+%m-%d %H:%M') on $(hostname)"
FREE=$(df -Pk $L | tail -1 | awk '{print $4}'); [ "$FREE" -gt 25000000 ] || { echo "need 25 GB free on $L"; exit 3; }
md5sum $D/RSAR.tar | tee $OUT/rsar_tar.md5
tar -xf $D/RSAR.tar -C $L/rsar
tar -xf $D/sardet100k-yolo.tar -C $L/sd images 2>&1 | grep -v "unknown extended header" || true
tar -xf $D/sardet100k-resplit.tar -C $L/sd test_images; mv $L/sd/test_images $L/sd/images/test
echo "unpacked $(date '+%H:%M'): RSAR $(ls $L/rsar/*/images | wc -l), SARDet $(ls $L/sd/images/* | wc -l) (counts include headers)"
python -u ~/sbatch-staging/rsar/match.py $L/rsar $L/sd $OUT
echo "done $(date '+%m-%d %H:%M')"
