#!/bin/bash
#SBATCH --job-name=persistence
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=gisr115048
#SBATCH --output=persistence_%j.out
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
export PYTHONUNBUFFERED=1
cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
python scripts/persistence_baseline.py
