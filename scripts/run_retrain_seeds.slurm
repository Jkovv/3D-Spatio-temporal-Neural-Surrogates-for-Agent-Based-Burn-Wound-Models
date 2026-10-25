#!/bin/bash
#SBATCH --job-name=retrain_seeds
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=retrain_seeds_%j.out
# Retrains DeepONet and U-Net for seeds 1 and 100 (IL-8, IL-10) with the seed-42 hyperparameters,
# into models/<model>_3d_retrain; only the weight files are copied to models/<model>_3d, the result
# JSONs there are left unchanged. Then the rollout is rerun.
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE/scripts
for MODEL in deeponet unet; do
    OUT=$BASE/models/${MODEL}_3d_retrain
    mkdir -p $OUT
    cp -n $BASE/models/${MODEL}_3d/res_il8_run_0062_50_42.json $BASE/models/${MODEL}_3d/res_il10_run_0062_50_42.json $OUT/
    for CYT in il8 il10; do
        for SEED in 1 100; do
            echo "$MODEL $CYT seed $SEED"
            python $MODEL/train_${MODEL}_3d.py --cytokine $CYT --seed $SEED --run run_0062 --grid 50 \
                --data $BASE/preprocessed_3d --out $OUT
            cp $OUT/weights_${CYT}_run_0062_50_${SEED}.weights.h5 $BASE/models/${MODEL}_3d/
        done
    done
done
cd $BASE
python scripts/rollout_3d.py --data $BASE/preprocessed_3d --models $BASE/models --out $BASE/models/rollout_3d --start 80
echo "DONE"
