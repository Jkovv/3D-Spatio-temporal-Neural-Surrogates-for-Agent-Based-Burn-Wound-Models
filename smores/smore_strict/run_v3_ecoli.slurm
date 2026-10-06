#!/bin/bash
#SBATCH --job-name=smore_v3_ecoli
#SBATCH --partition=rome
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=gisr115048
#SBATCH --output=smore_v3_ecoli_%j.out
# External E. coli sweep, v3: per-fold admissible sets and genome classification stored, and the positive
# control repeated without the 4.63 Mb strain, whose curves end at 24 h instead of 48.5 h (the plateau and
# midpoint of its synthetic curves are not observable, which is why the genome was not recovered in v2).
# Quick test first:  sbatch --export=ALL,FOLDS=3 run_v3_ecoli.slurm
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
ECOLI=$SM/external_data
CURVES=$ECOLI/table_S2_curves.xlsx
DESIGN=$ECOLI/41598_2025_32144_MOESM4_ESM.xlsx
MEDIA=$ECOLI/41598_2025_32144_MOESM2_ESM.xlsx
OUT=$SM/results_v3${FOLDS:+_test}
mkdir -p $OUT; cd $SM
F=${FOLDS:+--folds $FOLDS}
P="--procs $SLURM_CPUS_PER_TASK $F --sigma-mode paper --n-sd 2"
CACHE=$OUT/ecoli_arrays.npz
[ -f $SM/results/ecoli_arrays.npz ] && cp -n $SM/results/ecoli_arrays.npz $CACHE || true
if [ ! -f $CACHE ]; then SRC="--curves $CURVES --design $DESIGN --media $MEDIA --save-npz $CACHE"; else SRC="--from-npz $CACHE"; fi
echo "=== [1/5] real curves, design grid ==="
$PY smore_strict/ecoli_strict.py $SRC $P --mode real --out $OUT/ecoli_real.json
echo "=== [2/5] positive control, all five strains (as in v2, with the identifiability diagnostic) ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE $P --mode synthetic --out $OUT/ecoli_synthetic.json
echo "=== [3/5] positive control without the 4.63 Mb strain (24 h window) ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE $P --mode synthetic --drop-genome 4.63 --out $OUT/ecoli_synthetic_4strains.json
echo "=== [4/5] real curves, continuous box (for comparison) ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE $P --mode real --continuous --out $OUT/ecoli_real_continuous.json
echo "=== [5/5] real curves, design grid, sigma 'quadrature' ==="
$PY smore_strict/ecoli_strict.py --from-npz $CACHE --procs $SLURM_CPUS_PER_TASK $F --sigma-mode quadrature --n-sd 2 --mode real --out $OUT/ecoli_real_quadrature.json
echo "=== DONE ==="
