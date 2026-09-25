#!/bin/bash
#SBATCH --job-name=calibration
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=gisr115048
#SBATCH --output=calibration_%j.out

set -euo pipefail

BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
cd $BASE/smores/smore

echo "=== [1/3] topk10 (all 10 params) ==="
$PY run_calibration.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --top-k 10 \
    --n-saltelli 1024 \
    --out $BASE/smores/results/calibration_results_topk10.json

echo "=== [2/3] topk5 by Sobol ==="
$PY run_calibration.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --params sigmoidb keil8 init_ec km1il6 km2il10 \
    --n-saltelli 1024 \
    --out $BASE/smores/results/calibration_results_topk5_sobol.json

echo "=== [3/3] topk5 identifiable ==="
$PY run_calibration.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --params sigmoidb keil8 km1il6 km2il10 km2tgf \
    --n-saltelli 1024 \
    --out $BASE/smores/results/calibration_results_topk5_identifiable.json

echo "=== DONE ALL 3 ==="
