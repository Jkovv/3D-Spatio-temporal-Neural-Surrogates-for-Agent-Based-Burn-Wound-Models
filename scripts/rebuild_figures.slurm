#!/bin/bash
#SBATCH --job-name=rebuild_figs
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=rebuild_figs_%j.out

set -e
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
cd "$BASE"

module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

python -m py_compile scripts/figures.py scripts/fig_seeddots.py scripts/assemble_panels.py
echo "== syntax OK"

echo "== figures.py"
python scripts/figures.py --models-root models --prep-root preprocessed_3d \
    --sweep-root smores/results --grid 50 --out figures_3d

echo "== fig_seeddots.py"
python scripts/fig_seeddots.py

echo "== assemble_panels.py"
python scripts/assemble_panels.py --in figures_3d --out figures

echo "== done"; ls -lt figures_3d | head -30; ls -lt figures
