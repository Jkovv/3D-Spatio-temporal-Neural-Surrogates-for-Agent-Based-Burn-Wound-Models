#!/bin/bash
#SBATCH --job-name=unet_seeds
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=unet_%j.out

# 3D U-Net surrogate training on the benchmark run (run_0062).
#   sbatch train_unet.slurm il8
#   sbatch train_unet.slurm il10
# Train/infer times go into the per-seed result JSON (cost table).

set -euo pipefail

CYTOKINE="${1:-il8}"
RUN="run_0062"
GRID=50

# ---- paths ----------------------------------------------------------------
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
SWEEP_OUT=$BASE/sweep/outputs
PREP=$BASE/preprocessed_3d
MODELS=$BASE/models/unet_3d
SCRIPTS=$BASE/scripts

echo "[unet] host=$(hostname)  cytokine=$CYTOKINE  run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module purge
module load 2023
# pinned TensorFlow+CUDA module, else the newest available one
if ! module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 2>/dev/null; then
    echo "[unet] pinned TF module not found, searching for an alternative..."
    TF_MOD=$(module -t avail 2>&1 | grep -i "TensorFlow.*CUDA" | sort -V | tail -1)
    if [ -z "$TF_MOD" ]; then
        echo "[unet] ERROR: no TensorFlow+CUDA module found. Run:"
        echo "        module load 2023 && module avail TensorFlow"
        echo "        then set the correct name in this script."
        exit 1
    fi
    echo "[unet] loading $TF_MOD"
    module load "$TF_MOD"
fi
echo "[unet] TensorFlow module: $(module -t list 2>&1 | grep -i tensorflow || echo '?')"

pip install --user --no-input optuna scikit-image 2>&1 | tail -2 || true

cd "$SCRIPTS"

# ---- preprocess the benchmark run (skipped if done) ----------------------
if [ ! -f "$PREP/$RUN/${GRID}x${GRID}x${GRID}/X_unet.npy" ]; then
    echo "[unet] preprocessing $RUN ..."
    python preprocessing/preprocess_3d.py \
        --sim-root "$SWEEP_OUT" \
        --out-root "$PREP" \
        --grid "$GRID" \
        --runs "$RUN"
else
    echo "[unet] preprocessed data for $RUN exists, skipping."
fi

# ---- extra seeds (reuse the seed-42 hyperparameters) ----------------------
for SEED in 2 7 13 256 1000 2026 9999; do
    echo "[unet] === training $CYTOKINE seed=$SEED ==="
    python unet/train_unet_3d.py \
        --cytokine "$CYTOKINE" \
        --seed "$SEED" \
        --run "$RUN" \
        --grid "$GRID" \
        --data "$PREP" \
        --out "$MODELS"
done

echo "[unet] DONE. Results in $MODELS/res_${CYTOKINE}_${RUN}_${GRID}_{42,1,100}.json"
echo "[unet] train/infer times are inside those JSONs."
