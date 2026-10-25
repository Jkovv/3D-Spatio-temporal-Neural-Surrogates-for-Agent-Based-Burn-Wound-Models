#!/bin/bash
#SBATCH --job-name=loop_noclip
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=loop_noclip_%j.out
# Surrogate in the loop on the unclipped scale: DeepONet IL-8 retrained on preprocessed_3d_noclip
# (seeds 42, 1, 100), predictions at every sweep point, then SMoRe ParS. Needs run_prep_noclip.slurm first.
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
SM=$BASE/smores
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

cd $BASE/scripts
for SEED in 42 1 100; do
    echo "training il8 seed $SEED (unclipped)"
    python deeponet/train_deeponet_3d.py --cytokine il8 --seed $SEED --run run_0062 --grid 50 \
        --data $BASE/preprocessed_3d_noclip --out $BASE/models/deeponet_3d_noclip
done

cd $SM/smore
for SEED in 42 1 100; do
    echo "surrogate predictions at the sweep points, seed $SEED"
    python run_calibration_surrogate.py --sim-root $BASE/sweep/outputs --infer-root $BASE/preprocessed_3d_infer_noclip \
        --weights-dir $BASE/models/deeponet_3d_noclip --manifest $SM/manifest.json \
        --model deeponet --run-tag run_0062 --seed $SEED --grid 50 --cytokines il8 \
        --out $SM/results/calibration_surrogate_il8_noclip_seed${SEED}.json
done

cd $SM
for SEED in 42 1 100; do
    echo "loop, seed $SEED"
    $PY smore_strict/loop_strict.py --trajectories-npz $SM/results/calibration_surrogate_il8_noclip_seed${SEED}_trajectories.npz \
        --manifest $SM/manifest.json --procs $SLURM_CPUS_PER_TASK --out $SM/results_v3/loop_noclip_seed${SEED}.json
done
echo "DONE"
