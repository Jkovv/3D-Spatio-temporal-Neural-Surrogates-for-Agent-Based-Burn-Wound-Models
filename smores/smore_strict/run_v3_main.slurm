#!/bin/bash
#SBATCH --job-name=smore_v3_main
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_v3_main_%j.out
# SMoRe ParS on the ABM sweep, v3: chains run until split R-hat <= 1.1, per-fold region summaries,
# every-5th-frame check. Writes to results_v3/.
# Quick test first:  sbatch --export=ALL,FOLDS=3 run_v3_main.slurm   (a few minutes)
# Full run:          sbatch run_v3_main.slurm
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
OUT=$SM/results_v3${FOLDS:+_test}
mkdir -p $OUT; cd $SM
F=${FOLDS:+--folds $FOLDS}
P="--replicates-npz $SM/results/replicates.npz --manifest $SM/manifest.json --procs $SLURM_CPUS_PER_TASK $F"
# reuse the cached v2 SM fits
for s in smore_main smore_top5 smore_nsd1 smore_quadrature; do [ -f $SM/results/$s.json.fits.json ] && cp -n $SM/results/$s.json.fits.json $OUT/ || true; done

echo "=== [1/5] main: 10 parameters, sigma 'paper', 2 SD ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 2 --out $OUT/smore_main.json
echo "=== [2/5] temporal-correlation check: every 5th frame (21 of 101) ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 2 --thin 5 --out $OUT/smore_main_thin5.json
echo "=== [3/5] SM misfit added in quadrature ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode quadrature --n-sd 2 --out $OUT/smore_quadrature.json
echo "=== [4/5] top five Sobol parameters ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 2 --params sigmoidb km1il6 km2il10 init_ec keil8 --out $OUT/smore_top5.json
echo "=== [5/5] bounds widened by 1 SD (only containment and non-emptiness are used, so chains are kept short) ==="
$PY smore_strict/smore_pars_strict.py $P --sigma-mode paper --n-sd 1 --mcmc-max-steps 50 --out $OUT/smore_nsd1.json
# loop results are the same as in v2
cp -n $SM/results/loop_seed*.json $OUT/ 2>/dev/null || true
echo "=== DONE ==="
