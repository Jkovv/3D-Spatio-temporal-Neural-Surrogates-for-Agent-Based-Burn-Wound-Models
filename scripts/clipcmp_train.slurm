#!/bin/bash
#SBATCH --job-name=clipcmp_train
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=clipcmp_train_%j.out
# Clipped vs unclipped normalisation, step 2: one model on one normalisation.
#   sbatch --export=ALL,MODEL=deeponet,SCALE=clip  clipcmp_train.slurm
#   MODEL = deeponet | unet, SCALE = clip | noclip, CYTS (default "il8")
# Seed 42 runs Optuna (same search space as the main benchmark), seeds 1 and 100 reuse its HP.
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${MODEL:?set MODEL}"; : "${SCALE:?set SCALE}"; CYTS="${CYTS:-il8}"
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK=/gpfs/scratch1/shared/jkowalczuk/clipcmp
OUT=$BASE/models/clip_comparison/${MODEL}_${SCALE}
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
mkdir -p $OUT
cd $BASE/scripts
for CYT in $CYTS; do
    for SEED in 42 1 100; do
        echo "=== $MODEL $SCALE $CYT seed $SEED ==="
        python $MODEL/train_${MODEL}_3d.py --cytokine $CYT --seed $SEED --run run_0062 --grid 50 \
            --data $WORK/prep_$SCALE --out $OUT
    done
done
echo "=== DONE ==="
