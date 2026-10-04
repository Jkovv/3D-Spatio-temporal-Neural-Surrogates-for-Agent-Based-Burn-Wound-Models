#!/bin/bash
#SBATCH --job-name=smore_v2_ecoli
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_v2_ecoli_%j.out
# External E. coli sweep: real curves and the positive control on the design grid, plus the
# continuous-box search of v1 for comparison.  Set ECOLI to the folder with the three
# supplementary spreadsheets of Gong & Ying and check the three file names below.
# Quick test first:  sbatch --export=ALL,FOLDS=3 run_v2_ecoli.slurm
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
ECOLI=$SM/external_data
CURVES=$ECOLI/table_S2_curves.xlsx
DESIGN=$ECOLI/41598_2025_32144_MOESM4_ESM.xlsx
MEDIA=$ECOLI/41598_2025_32144_MOESM2_ESM.xlsx
OUT=$SM/results${FOLDS:+_test}
mkdir -p $OUT; cd $SM
F=${FOLDS:+--folds $FOLDS}
P="--procs $SLURM_CPUS_PER_TASK $F --sigma-mode paper --n-sd 2"
CACHE=$OUT/ecoli_arrays.npz
if [ ! -f $CACHE ]; then SRC="--curves $CURVES --design $DESIGN --media $MEDIA --save-npz $CACHE"; else SRC="--from-npz $CACHE"; fi
echo "=== [1/4] real curves, design grid ==="
$PY smore_strict/ecoli_strict.py $SRC $P --mode real --out $OUT/ecoli_real.json
echo "=== [2/4] positive control, design grid ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE $P --mode synthetic --out $OUT/ecoli_synthetic.json
echo "=== [3/4] real curves, continuous box (v1 search space, for comparison) ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE $P --mode real --continuous --out $OUT/ecoli_real_continuous.json
echo "=== [4/4] real curves, design grid, sigma 'quadrature' (same robustness check as on the ABM) ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE --procs $SLURM_CPUS_PER_TASK $F --sigma-mode quadrature --n-sd 2 --mode real --out $OUT/ecoli_real_quadrature.json
echo "=== DONE ==="
