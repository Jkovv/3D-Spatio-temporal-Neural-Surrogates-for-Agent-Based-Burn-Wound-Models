#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=gisr115048
#SBATCH --output=figures_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
python -c "import matplotlib, skimage" 2>/dev/null || pip install --user --no-input matplotlib scikit-image
cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
echo "=== [1/2] figures.py ==="
python scripts/figures.py --models-root models --prep-root preprocessed_3d --sweep-root smores/results --grid 50 --out figures_3d
echo "=== [2/2] assemble_panels.py ==="
python scripts/assemble_panels.py --in figures_3d --out figures
echo "=== DONE ALL ==="
