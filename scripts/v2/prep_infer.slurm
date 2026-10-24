#!/bin/bash
#SBATCH --job-name=v2_prep_infer
#SBATCH --account=gisr115048
#SBATCH --partition=rome
#SBATCH --array=1-100
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=v2_prep_infer_%A_%a.out
# New sweep realisation, step 3: every sweep point on the frozen scale of run_0062, clipped and unclipped.
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK="${WORK:-/gpfs/scratch1/shared/jkowalczuk/sweep_v2}"
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
RUN=$(printf "run_%04d" $SLURM_ARRAY_TASK_ID)
cd $BASE/scripts
for S in clip noclip; do
    $PY preprocessing/preprocess_3d.py --sim-root $WORK/outputs --out-root $WORK/infer_$S --grid 50 --runs $RUN \
        --scale-from $WORK/prep_$S/run_0062/50x50x50/metadata.json
done
echo "=== DONE $RUN ==="
