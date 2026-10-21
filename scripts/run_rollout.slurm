#!/bin/bash
#SBATCH --job-name=rollout_3d
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=rollout_%j.out
# Autoregressive rollout of both surrogates (IL-8, IL-10, seeds 1/42/100) over the test window.
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE
python scripts/rollout_3d.py --data $BASE/preprocessed_3d --models $BASE/models --out $BASE/models/rollout_3d --start 80
echo "=== DONE ==="
