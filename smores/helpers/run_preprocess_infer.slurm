#!/bin/bash
#SBATCH --job-name=prep_infer
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=gisr115048
#SBATCH --output=prep_infer_%j.out
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SCALE=$BASE/preprocessed_3d/run_0062/50x50x50/metadata.json
cd $BASE/scripts
n_done=0
for i in $(seq 1 100); do
    RUN=$(printf "run_%04d" $i)
    if [ -d "$BASE/preprocessed_3d_infer/$RUN/50x50x50" ]; then
        continue
    fi
    echo "=== preprocessing $RUN ==="
    $PY preprocessing/preprocess_3d.py \
        --sim-root $BASE/sweep/outputs \
        --out-root $BASE/preprocessed_3d_infer \
        --grid 50 \
        --runs $RUN \
        --scale-from $SCALE && n_done=$((n_done+1)) || echo "  [FAIL] $RUN"
done
echo "=== DONE: preprocessed $n_done runs ==="
