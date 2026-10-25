#!/bin/bash
#SBATCH --job-name=diag_late
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=diag_late_%j.out
# Per-frame diagnostics of the one-step predictions in the test window (default: IL-10 DeepONets).
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE
python scripts/diag_late_window.py --kind deeponet --cytokine il10 --seeds 1 42 100
python scripts/diag_late_window.py --kind unet --cytokine il10 --seeds 1 42 100
echo "DONE"
