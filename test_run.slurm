#!/bin/bash
#SBATCH --job-name=cc3d_testrun
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=testrun_%j.out

# ============================================================================
# ONE test run (run_0001 only) to confirm CC3D actually launches and produces
# datafiles/mean_concentration.txt before committing to the full sweep.
#
#   sbatch test_run.slurm
#
# On success you'll see the output file listed at the end. THEN run the full
# sweep (run_sweep.py --mode slurm + sbatch sweep/sweep_array.sh).
# ============================================================================

set -euo pipefail

# ---- edit if paths differ -------------------------------------------------
SCRATCH=/gpfs/scratch1/shared/jkowalczuk
CONDA_ROOT=$SCRATCH/miniconda3
ENV_NAME=cc3d
BASE=$SCRATCH/surrogates/burns/combi3d
# ---------------------------------------------------------------------------

CC3DPY=$CONDA_ROOT/envs/$ENV_NAME/bin/python
SIM=$BASE/combi3D

source "$CONDA_ROOT/bin/activate" "$ENV_NAME"
cd "$SIM"

echo "[test] verify pipeline (no CC3D)..."
python verify.py

echo "[test] generating manifest..."
python setup_runs.py --n-runs 10 --seed 42

echo "[test] staging run dirs (slurm mode just stages; we launch run_0001 by hand)..."
python run_sweep.py --manifest manifest.json --mode slurm \
    --cc3d-python "$CC3DPY" \
    --cc3d-file "$SIM/combi3D.cc3d" \
    --runs-dir "$BASE/sweep/runs" \
    --out "$BASE/sweep/outputs" \
    --work "$BASE/sweep"

echo "[test] launching run_0001 directly (TIMED for speedup)..."
RUNDIR=$BASE/sweep/runs/run_0001
OUTDIR=$BASE/sweep/outputs/run_0001
mkdir -p "$OUTDIR"

# ── ABM wall-clock timing ────────────────────────────────────────────────────
# T_ABM is the ONLY missing number for the speedup / break-even table in the
# paper and thesis:
#     speedup = T_ABM / T_infer
#     N*      = T_train / (T_ABM - T_infer)
# Measure ONLY the CC3D run here (not verify/staging), on a clean node, so the
# figure is comparable to the thesis timing tables. Record grid + step count so
# the measurement is reproducible.
ABM_GRID=$(python -c "import sys; sys.path.insert(0,'$SIM'); from params_grid import L; print(L)")
ABM_STEPS=${COMBI3D_STEPS:-1000001}
TIMER_FILE="$OUTDIR/abm_walltime.txt"

echo "[test] T_ABM measurement: grid=${ABM_GRID}^3, Steps=${ABM_STEPS}, cpus=${SLURM_CPUS_PER_TASK:-4}"
_t_start=$(date +%s.%N)

"$CC3DPY" -m cc3d.run_script \
    --input="$RUNDIR/combi3D.cc3d" \
    --output-dir="$OUTDIR"

_t_end=$(date +%s.%N)
_elapsed=$(python -c "print(f'{${_t_end} - ${_t_start}:.2f}')")

# Persist the measurement in a machine- and human-readable form.
{
  echo "T_ABM_seconds=${_elapsed}"
  echo "grid=${ABM_GRID}"
  echo "steps=${ABM_STEPS}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-4}"
  echo "partition=${SLURM_JOB_PARTITION:-rome}"
  echo "run_id=run_0001"
  echo "slurm_job=${SLURM_JOB_ID:-NA}"
  echo "measured_at=$(date -Iseconds)"
} > "$TIMER_FILE"

echo "[test] ===================================================="
echo "[test]  T_ABM = ${_elapsed} s   (one ${ABM_GRID}^3 run, 100 h)"
echo "[test]  saved -> $TIMER_FILE"
echo "[test]  speedup vs surrogate = T_ABM / T_infer"
echo "[test]  (thesis T_infer: U-Net ~1.8 s, DeepONet ~35 s)"
echo "[test] ===================================================="
# ─────────────────────────────────────────────────────────────────────────────

cp "$RUNDIR/Simulation/params.json" "$OUTDIR/params.json"

echo ""
echo "[test] checking outputs..."
MC=$(find "$OUTDIR" -name "mean_concentration.txt" | head -1)
if [ -n "$MC" ]; then
    echo "  [ok]  found $MC"
    echo "  first lines:"
    head -3 "$MC" | sed 's/^/    /'
    echo "  [ok]  params.json: $(ls $OUTDIR/params.json)"
    echo ""
    echo "[test] SUCCESS -- CC3D works. Now run the full sweep:"
    echo "  sbatch $BASE/sweep/sweep_array.sh"
else
    echo "  [FAIL] no mean_concentration.txt produced."
    echo "  Check the CC3D output tree:"
    find "$OUTDIR" -maxdepth 3 -type d | sed 's/^/    /'
    exit 1
fi
