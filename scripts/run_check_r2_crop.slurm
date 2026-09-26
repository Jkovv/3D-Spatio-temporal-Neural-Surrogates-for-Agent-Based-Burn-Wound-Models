#!/bin/bash
#SBATCH --job-name=check_r2_crop
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=gisr115048
#SBATCH --output=check_r2_crop_%j.out
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
export PYTHONUNBUFFERED=1
cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
python scripts/check_r2_crop.py
