#!/bin/bash
#SBATCH --job-name=v2_eval
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=v2_eval_%j.out
# New sweep realisation, step 7: clip comparison, rollout and the one-step baselines on the new run_0062.
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK="${WORK:-/gpfs/scratch1/shared/jkowalczuk/sweep_v2}"
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE
python scripts/clip_ceiling.py --work $WORK --out models/v2/clip_ceiling.json
python scripts/clip_comparison.py --work $WORK --models models/v2 --cytokines il8
# rollout of the clipped-protocol networks (both cytokines), as in the main text
mkdir -p $WORK/rollout_models
ln -sfn $BASE/models/v2/deeponet_clip $WORK/rollout_models/deeponet_3d
ln -sfn $BASE/models/v2/unet_clip $WORK/rollout_models/unet_3d
python scripts/rollout_3d.py --data $WORK/prep_clip --models $WORK/rollout_models --out models/v2/rollout_3d --start 80
# persistence and linear extrapolation on the clipped protocol (reads preprocessed_3d/ relative to the cwd)
mkdir -p $WORK/pb
ln -sfn $WORK/prep_clip $WORK/pb/preprocessed_3d
cd $WORK/pb
python $BASE/scripts/persistence_baseline.py
cp models/persistence_baseline_run_0062_50.json $BASE/models/v2/
echo "DONE"
