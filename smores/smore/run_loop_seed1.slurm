#!/bin/bash
#SBATCH --job-name=loop_s1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=gisr115048
#SBATCH --output=loop_seed1_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/smore
python run_calibration_surrogate.py \
    --sim-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/sweep/outputs \
    --infer-root /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/preprocessed_3d_infer \
    --weights-dir /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/models/deeponet_3d \
    --manifest /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/manifest.json \
    --model deeponet --run-tag run_0062 --seed 1 --grid 50 \
    --cytokines il8 \
    --out /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models/smores/results/calibration_surrogate_il8_seed1.json
echo "=== DONE seed 1 ==="
