#!/bin/bash
#SBATCH --job-name=deeponet_3d
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=deeponet_%j.out

# ============================================================================
# DeepONet 3D surrogate training on the benchmark run (run_0062).
#
#   sbatch train_deeponet.slurm il8      # trains IL-8 on seeds 42,1,100
#   sbatch train_deeponet.slurm il10     # trains IL-10
#
# What it does:
#   1. preprocess run_0062 -> .npy tensors (only if not already done)
#   2. train seed 42 first  (runs Optuna, writes HP json)
#   3. train seeds 1 and 100 (reuse seed-42 HP for stability check)
#
# GPU: 1x A100 (partition gpu = 18 cores + 1 GPU, 128 SBU/hr).
# Records train/infer times in the per-seed result JSON -> feeds the cost table.
#
# NOTE ON PYTHON ENV: surrogate training needs TensorFlow + GPU, which is a
# DIFFERENT environment from the CC3D 'cc3d' conda env used for the sweep.
# We use Snellius's module TensorFlow build (CUDA 12.1), matching the paper.
# ============================================================================

set -euo pipefail

CYTOKINE="${1:-il8}"          # first arg: cytokine (default il8)
RUN="run_0062"                # benchmark run (see helpers/benchmark_run_justification.txt)
GRID=50

# ---- paths ----------------------------------------------------------------
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns
SWEEP_OUT=$BASE/sweep/outputs
PREP=$BASE/preprocessed_3d
MODELS=$BASE/models/deeponet_3d
SCRIPTS=$BASE/scripts
# ---------------------------------------------------------------------------

echo "[deeponet] host=$(hostname)  cytokine=$CYTOKINE  run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- environment: TensorFlow + CUDA via modules ---------------------------
# Adjust the exact module string to what `module avail TensorFlow` shows on
# Snellius. This one matches the paper (TF 2.15.1, CUDA 12.1).
module purge
module load 2023
# Load TensorFlow (CUDA). The exact module string can vary between Snellius
# stack versions, so try the known name first, then fall back to whatever
# TensorFlow module is available. If none loads, the job stops with a clear
# message instead of failing cryptically later.
if ! module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 2>/dev/null; then
    echo "[deeponet] pinned TF module not found, searching for an alternative..."
    TF_MOD=$(module -t avail 2>&1 | grep -i "TensorFlow.*CUDA" | sort -V | tail -1)
    if [ -z "$TF_MOD" ]; then
        echo "[deeponet] ERROR: no TensorFlow+CUDA module found. Run:"
        echo "           module load 2023 && module avail TensorFlow"
        echo "           then set the correct name in this script."
        exit 1
    fi
    echo "[deeponet] loading $TF_MOD"
    module load "$TF_MOD"
fi
echo "[deeponet] TensorFlow module: $(module -t list 2>&1 | grep -i tensorflow || echo '?')"

# extra python deps not in the TF module (optuna, skimage, SALib not needed here)
pip install --user --no-input optuna scikit-image 2>&1 | tail -2 || true

cd "$SCRIPTS"

# ---- 1. preprocess the benchmark run (idempotent) -------------------------
if [ ! -f "$PREP/$RUN/${GRID}x${GRID}x${GRID}/X_branch.npy" ]; then
    echo "[deeponet] preprocessing $RUN ..."
    python preprocessing/preprocess_3d.py \
        --sim-root "$SWEEP_OUT" \
        --out-root "$PREP" \
        --grid "$GRID" \
        --runs "$RUN"
else
    echo "[deeponet] preprocessed data for $RUN exists, skipping."
fi

# ---- 2+3. train seed 42 FIRST (Optuna), then 1 and 100 --------------------
for SEED in 42 1 100; do
    echo "[deeponet] === training $CYTOKINE seed=$SEED ==="
    python deeponet/train_deeponet_3d.py \
        --cytokine "$CYTOKINE" \
        --seed "$SEED" \
        --run "$RUN" \
        --grid "$GRID" \
        --data "$PREP" \
        --out "$MODELS"
done

echo "[deeponet] DONE. Results in $MODELS/res_${CYTOKINE}_${RUN}_${GRID}_{42,1,100}.json"
echo "[deeponet] train/infer times are inside those JSONs (train_time_seconds, pred_time_seconds)."
