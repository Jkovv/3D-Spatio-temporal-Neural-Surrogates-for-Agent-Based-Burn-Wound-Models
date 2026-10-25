#!/bin/bash
#SBATCH --job-name=clipcmp_eval
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=clipcmp_eval_%j.out
# Clipped vs unclipped normalisation, step 3: every trained model against the unclipped truth,
# one step ahead and in a rollout, plus persistence and linear extrapolation. CYTS (default "il8").
set -euo pipefail
export PYTHONUNBUFFERED=1
CYTS="${CYTS:-il8}"
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE
python scripts/clip_comparison.py --work /gpfs/scratch1/shared/jkowalczuk/clipcmp \
    --models $BASE/models/clip_comparison --cytokines $CYTS
echo "DONE"
