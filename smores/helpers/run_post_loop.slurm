#!/bin/bash
#SBATCH --job-name=post_loop
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=gisr115048
#SBATCH --output=post_loop_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/helpers

echo "=== [1/2] predictive_consistency ==="
python predictive_consistency.py \
    --calibration /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/results/calibration_surrogate_il8.json \
    --sim-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/sweep/outputs \
    --infer-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/preprocessed_3d_infer \
    --manifest /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/manifest.json \
    --smore-dir /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/smore \
    --cytokine il8 --grid 50 \
    --out /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/results/predictive_consistency.json

echo "=== [2/2] decompose_observable_gap ==="
python decompose_observable_gap.py \
    --sim-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/sweep/outputs \
    --infer-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/preprocessed_3d_infer \
    --manifest /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/manifest.json \
    --smore-dir /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/smore \
    --cytokine il8 --grid 50 \
    --out /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/results/observable_gap_decomposition.json

echo "=== DONE ALL ==="
