#!/bin/bash
#SBATCH --job-name=v2_loop
#SBATCH --account=gisr115048
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=v2_loop_%j.out
# New sweep realisation, step 6: surrogate in the loop (SCALE = clip | noclip, SEED).
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SCALE:?set SCALE}"; : "${SEED:?set SEED}"
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
mkdir -p $BASE/smores/results_v3/v2
cd $BASE/smores
$PY smore_strict/loop_strict.py --trajectories-npz results/v2/calibration_surrogate_il8_${SCALE}_seed${SEED}_trajectories.npz \
    --manifest manifest.json --procs $SLURM_CPUS_PER_TASK --out results_v3/v2/loop_${SCALE}_seed${SEED}.json
echo "=== DONE ==="
