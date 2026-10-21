#!/bin/bash
#SBATCH --job-name=smore_lc
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_lc_%j.out
# SMoRe ParS on nested subsamples of 60, 70, 80 and 90 sweep points (same settings as smore_main).
# The SM fits of the main run are reused; only the bound surfaces and regions are recomputed.
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
OUT=$SM/results_v3
cd $SM
P="--replicates-npz $SM/results/replicates.npz --manifest $SM/manifest.json --procs $SLURM_CPUS_PER_TASK --sigma-mode paper --n-sd 2"
for n in 60 70 80 90; do
  cp -n $OUT/smore_main.json.fits.json $OUT/smore_main_n${n}.json.fits.json
  echo "=== $n sweep points ==="
  $PY smore_strict/smore_pars_strict.py $P --subsample $n --out $OUT/smore_main_n${n}.json
done
echo "=== DONE ==="
