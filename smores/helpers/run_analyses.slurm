#!/bin/bash
#SBATCH --job-name=analyses
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=gisr115048
#SBATCH --output=analyses_%j.out
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
cd $BASE/smores/helpers

echo "=== [1/3] emulator_cv ==="
$PY emulator_cv.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --smore-dir $BASE/smores/smore \
    --out-tex $BASE/smores/results/appendix_emulator.tex \
    --out-csv $BASE/smores/results/emulator_cv.csv \
    --folds 5

echo "=== [2/3] sobol_convergence ==="
$PY sobol_convergence.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --subsamples 60 70 80 90 100 \
    --n-saltelli 512

echo "=== [3/3] ridge ==="
$PY init_ec_keil8_ridge.py \
    --manifest $BASE/smores/manifest.json \
    --sim-root $BASE/sweep/outputs \
    --observable il8_final \
    --out $BASE/smores/results/ridge_init_ec_keil8.json

echo "=== DONE ALL ==="
