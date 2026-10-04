#!/bin/bash
#SBATCH --job-name=smore_v2_loop
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_v2_loop_%j.out
# Surrogate-in-the-loop (IL-8) with error propagation and the matched-error control, three training seeds.
# Quick test first:  sbatch --export=ALL,FOLDS=3 run_v2_loop.slurm
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
OUT=$SM/results${FOLDS:+_test}
mkdir -p $OUT; cd $SM
F=${FOLDS:+--folds $FOLDS}
for S in 1 42 100; do
  if [ "$S" = "42" ]; then NPZ=$SM/results/calibration_surrogate_il8_trajectories.npz
  else NPZ=$SM/results/calibration_surrogate_il8_seed${S}_trajectories.npz; fi
  echo "=== loop, surrogate seed $S ==="
  $PY smore_strict/loop_strict.py --trajectories-npz $NPZ --manifest $SM/manifest.json \
      --procs $SLURM_CPUS_PER_TASK $F --out $OUT/loop_seed${S}.json
done
echo "=== DONE ==="
