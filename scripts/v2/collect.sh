#!/bin/bash
# Copy the results of the chain (models/v2, smores/results/v2, smores/results_v3/v2) to the
# directories the paper's tables and figures are built from, and rebuild the tables.
# Run on a login node after eval.slurm and all loop jobs have finished:  bash scripts/v2/collect.sh
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
cd $BASE

for f in models/v2/clip_comparison.json models/v2/rollout_3d/rollout_unet_il10_run_0062_50_100.json models/v2/persistence_baseline_run_0062_50.json \
         smores/results_v3/v2/loop_clip_seed100.json smores/results_v3/v2/loop_noclip_seed100.json; do
    [ -f $f ] || { echo "missing $f: the chain has not finished"; exit 1; }
done

mkdir -p models/deeponet_3d models/unet_3d models/rollout_3d models/clip_comparison/deeponet_noclip models/clip_comparison/unet_noclip
cp models/v2/deeponet_clip/res_*.json models/deeponet_3d/
cp models/v2/unet_clip/res_*.json models/unet_3d/
cp models/v2/rollout_3d/*.json models/rollout_3d/
cp models/v2/clip_comparison.json models/v2/clip_ceiling.json models/clip_comparison/
cp models/v2/deeponet_noclip/res_*.json models/clip_comparison/deeponet_noclip/
cp models/v2/unet_noclip/res_*.json models/clip_comparison/unet_noclip/
cp models/v2/persistence_baseline_run_0062_50.json models/

for SEED in 42 1 100; do
    cp smores/results/v2/calibration_surrogate_il8_clip_seed${SEED}.json smores/results/calibration_surrogate_il8_seed${SEED}.json
    cp smores/results/v2/calibration_surrogate_il8_clip_seed${SEED}_trajectories.npz smores/results/calibration_surrogate_il8_seed${SEED}_trajectories.npz
    cp smores/results/v2/calibration_surrogate_il8_noclip_seed${SEED}.json smores/results/
    cp smores/results/v2/calibration_surrogate_il8_noclip_seed${SEED}_trajectories.npz smores/results/
    cp smores/results_v3/v2/loop_clip_seed${SEED}.json smores/results_v3/loop_seed${SEED}.json
    cp smores/results_v3/v2/loop_noclip_seed${SEED}.json smores/results_v3/
done

# tables (numpy and matplotlib only)
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
python scripts/forward_numbers.py
python scripts/rollout_summary.py
python scripts/clip_comparison.py --work . --models models/clip_comparison --table-only
cd smores
ECOLI=""; [ -f results_v3/ecoli_arrays.npz ] && ECOLI="--ecoli-npz results_v3/ecoli_arrays.npz"
python smore_strict/make_tables.py --results results_v3 --sobol results/sobol_repmean.json $ECOLI
echo "DONE"
