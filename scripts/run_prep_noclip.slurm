#!/bin/bash
#SBATCH --job-name=prep_noclip
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=gisr115048
#SBATCH --output=prep_noclip_%j.out
# Preprocessing without percentile clipping: the benchmark run with its own linear scale,
# then every sweep run on that frozen scale (for the surrogate-in-the-loop test).
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
cd $BASE/scripts
$PY preprocessing/preprocess_3d.py --sim-root $BASE/sweep/outputs --out-root $BASE/preprocessed_3d_noclip \
    --grid 50 --runs run_0062 --no-clip
SCALE=$BASE/preprocessed_3d_noclip/run_0062/50x50x50/metadata.json
for i in $(seq 1 100); do
    RUN=$(printf "run_%04d" $i)
    [ -d "$BASE/preprocessed_3d_infer_noclip/$RUN/50x50x50" ] && continue
    echo "=== $RUN ==="
    $PY preprocessing/preprocess_3d.py --sim-root $BASE/sweep/outputs --out-root $BASE/preprocessed_3d_infer_noclip \
        --grid 50 --runs $RUN --scale-from $SCALE || echo "  [FAIL] $RUN"
done
echo "=== DONE ==="
