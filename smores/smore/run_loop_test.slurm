#!/bin/bash
#SBATCH --job-name=loop_test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=gisr115048
#SBATCH --output=loop_test_%j.out
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd $BASE/smores/smore
echo "=== TEST loop seed=42 ==="
python run_calibration_surrogate.py \
    --sim-root $BASE/sweep/outputs \
    --infer-root $BASE/preprocessed_3d_infer \
    --weights-dir $BASE/models/deeponet_3d \
    --manifest $BASE/smores/manifest.json \
    --model deeponet --run-tag run_0062 --seed 42 --grid 50 \
    --cytokines il8 il10 \
    --out $BASE/smores/results/calibration_surrogate_il8_seed42.json
echo "=== TEST DONE ==="
