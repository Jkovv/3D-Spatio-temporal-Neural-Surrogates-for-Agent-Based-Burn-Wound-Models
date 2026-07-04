#!/bin/bash
#SBATCH --job-name=cc3d_timing
#SBATCH --partition=rome
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=timing_%j.out

# ============================================================================
# ABM RUNTIME BENCHMARK  ->  T_ABM (mean +/- SD)  [CPU / rome]
#
# This measures ONLY the ABM wall-clock, on the NEW sweep data you generate for
# this paper. It does NOT compute speedup here, because speedup also needs the
# surrogate's train/infer times, and those are produced separately when you
# train the surrogate FROM SCRATCH on this same new data (on GPU). Mixing in
# thesis surrogate numbers would be inconsistent — every timing in this paper's
# cost table comes from this new pipeline, nothing is carried over.
#
#   sbatch time_abm.slurm
#
# Output: sweep/timing_abm_summary.txt  (T_ABM mean +/- SD + individual times).
# Later, feed that T_ABM together with your NEW surrogate times into
# compute_speedup.py to fill the paper's cost table.
#
# Why mean +/- SD: the ABM is stochastic, so one run is a noisy sample. Timing
# N_REPEAT runs of the same parameter vector puts T_ABM on the same footing as
# the surrogate times (also reported as mean +/- SD over seeds).
#
# IMPORTANT: run this ALONE on the node (dedicated cpus), NOT concurrently with
# the sweep — a run competing for cores would inflate T_ABM.
# ============================================================================

set -euo pipefail

# ---- edit if paths differ -------------------------------------------------
SCRATCH=/gpfs/scratch1/shared/jkowalczuk
CONDA_ROOT=$SCRATCH/miniconda3
ENV_NAME=cc3d
BASE=$SCRATCH/surrogates/burns/combi3d
N_REPEAT=3                      # stochastic ABM -> report mean +/- SD
# ---------------------------------------------------------------------------

CC3DPY=$CONDA_ROOT/envs/$ENV_NAME/bin/python
SIM=$BASE/combi3D

source "$CONDA_ROOT/bin/activate" "$ENV_NAME"
cd "$SIM"

# Ensure a manifest + staged run_0001 exist (idempotent).
if [ ! -f manifest.json ]; then
    python setup_runs.py --n-runs 10 --seed 42
fi
if [ ! -d "$BASE/sweep/runs/run_0001" ]; then
    python run_sweep.py --manifest manifest.json --mode slurm \
        --cc3d-python "$CC3DPY" \
        --cc3d-file "$SIM/combi3D.cc3d" \
        --runs-dir "$BASE/sweep/runs" \
        --out "$BASE/sweep/outputs" \
        --work "$BASE/sweep"
fi

ABM_GRID=$(python -c "import sys; sys.path.insert(0,'$SIM'); from params_grid import L; print(L)")
ABM_STEPS=${COMBI3D_STEPS:-1000001}
RUNDIR=$BASE/sweep/runs/run_0001
TIMES_FILE="$BASE/sweep/timing_abm_times.txt"
SUMMARY="$BASE/sweep/timing_abm_summary.txt"
: > "$TIMES_FILE"

echo "[timing] grid=${ABM_GRID}^3, Steps=${ABM_STEPS}, repeats=${N_REPEAT}, cpus=${SLURM_CPUS_PER_TASK:-4}"

for i in $(seq 1 "$N_REPEAT"); do
    OUTDIR=$BASE/sweep/outputs/timing_rep_${i}
    mkdir -p "$OUTDIR"
    echo "[timing] repeat $i / $N_REPEAT ..."
    _t0=$(date +%s.%N)
    "$CC3DPY" -m cc3d.run_script \
        --input="$RUNDIR/combi3D.cc3d" \
        --output-dir="$OUTDIR"
    _t1=$(date +%s.%N)
    _dt=$(python -c "print(f'{${_t1} - ${_t0}:.2f}')")
    echo "$_dt" >> "$TIMES_FILE"
    echo "[timing]   repeat $i: ${_dt} s"
done

python - "$TIMES_FILE" "$SUMMARY" "$ABM_GRID" "$ABM_STEPS" <<'PYEOF'
import sys, statistics as st
times = [float(x) for x in open(sys.argv[1]) if x.strip()]
mean = st.mean(times)
sd   = st.pstdev(times) if len(times) > 1 else 0.0
lines = [
    f"T_ABM_mean_seconds = {mean:.2f}",
    f"T_ABM_sd_seconds   = {sd:.2f}",
    f"n_repeat           = {len(times)}",
    f"grid               = {sys.argv[3]}",
    f"steps              = {sys.argv[4]}",
    f"individual_times   = {', '.join(f'{t:.2f}' for t in times)}",
    "",
    "# T_ABM measured on rome/CPU on the NEW sweep data.",
    "# Next: train the surrogate from scratch on this data (GPU), read off its",
    "# train/infer times, then run compute_speedup.py with all three numbers.",
]
out = "\n".join(lines) + "\n"
open(sys.argv[2], "w").write(out)
print("\n[timing] ==================================================")
print(out)
print("[timing] saved -> " + sys.argv[2])
print("[timing] ==================================================")
PYEOF

echo "[timing] DONE. T_ABM is measured. Compute speedup once the surrogate is trained."
