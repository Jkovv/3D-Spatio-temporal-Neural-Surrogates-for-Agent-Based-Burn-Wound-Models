#!/bin/bash
#SBATCH --job-name=sobol_conv7
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=gisr115048
#SBATCH --output=sobol_conv7_%j.out
# Punkt 5: zbieznosc Sobola ze sledzeniem top-7 (zeby objac km2il10).
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
export PYTHONUNBUFFERED=1
cd $BASE/smores/helpers
cp -f sobol_convergence_report.txt sobol_convergence_report_top5.txt 2>/dev/null || true
$PY sobol_convergence.py \
    --sim-root $BASE/sweep/outputs \
    --manifest $BASE/smores/manifest.json \
    --subsamples 60 70 80 90 100 \
    --n-saltelli 512 \
    --top 7
echo "=== DONE ==="
