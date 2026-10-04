#!/bin/bash
#SBATCH --job-name=smore_v2_main
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_v2_main_%j.out
# SMoRe ParS on the ABM sweep with the corrected implementation (v2).
# Quick test first:  sbatch --export=ALL,FOLDS=3 run_v2_main.slurm   (a few minutes)
# Full run:          sbatch run_v2_main.slurm
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
OUT=$SM/results${FOLDS:+_test}
mkdir -p $OUT; cd $SM
F=${FOLDS:+--folds $FOLDS}
P="--replicates-npz $SM/results/replicates.npz --manifest $SM/manifest.json --procs $SLURM_CPUS_PER_TASK $F"

echo "=== [1/4] main: 10 parameters, sigma 'paper', 2 SD (Table recovery) ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 2 --out $OUT/smore_main.json
echo "=== [2/4] top five Sobol parameters (appendix) ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 2 --params sigmoidb km1il6 km2il10 init_ec keil8 --out $OUT/smore_top5.json
echo "=== [3/4] bounds widened by 1 SD (appendix) ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 1 --out $OUT/smore_nsd1.json
echo "=== [4/4] robustness: SM misfit added in quadrature for every cytokine ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode quadrature --n-sd 2 --out $OUT/smore_quadrature.json
echo "=== DONE ==="
