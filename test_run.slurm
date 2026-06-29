#!/bin/bash
#SBATCH --job-name=cc3d_testrun
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=testrun_%j.out

set -euo pipefail

SCRATCH=/gpfs/scratch1/shared/jkowalczuk
CONDA_ROOT=$SCRATCH/miniconda3
ENV_NAME=cc3d
BASE=$SCRATCH/surrogates/burns/combi3d

CC3DPY=$CONDA_ROOT/envs/$ENV_NAME/bin/python
SIM=$BASE/combi3D

source "$CONDA_ROOT/bin/activate" "$ENV_NAME"
cd "$SIM"

echo "testing the pipeline (no CC3D)..."
python verify.py

echo "[test] generating manifest..."
python setup_runs.py --n-runs 10 --seed 42

echo "[test] staging run dirs..."
python run_sweep.py --manifest manifest.json --mode slurm \
    --cc3d-python "$CC3DPY" \
    --cc3d-file "$SIM/combi3D.cc3d" \
    --runs-dir "$BASE/sweep/runs" \
    --out "$BASE/sweep/outputs" \
    --work "$BASE/sweep"

echo "[test] launching run_0001 directly..."
RUNDIR=$BASE/sweep/runs/run_0001
OUTDIR=$BASE/sweep/outputs/run_0001
mkdir -p "$OUTDIR"
"$CC3DPY" -m cc3d.run_script \
    --input="$RUNDIR/combi3D.cc3d" \
    --output-dir="$OUTDIR"

cp "$RUNDIR/Simulation/params.json" "$OUTDIR/params.json"

echo ""
echo "[test] checking outputs..."
MC=$(find "$OUTDIR" -name "mean_concentration.txt" | head -1)
if [ -n "$MC" ]; then
    echo "[ok] found $MC"
    echo "first lines:"
    head -3 "$MC" | sed 's/^/    /'
    echo "[ok] params.json: $(ls $OUTDIR/params.json)"
    echo ""
    echo "[test] SUCCESS - CC3D works. Now run the full sweep:"
    echo "sbatch $BASE/sweep/sweep_array.sh"
else
    echo "  [FAIL] no mean_concentration.txt produced."
    echo "Check the CC3D output tree:"
    find "$OUTDIR" -maxdepth 3 -type d | sed 's/^/    /'
    exit 1
fi
