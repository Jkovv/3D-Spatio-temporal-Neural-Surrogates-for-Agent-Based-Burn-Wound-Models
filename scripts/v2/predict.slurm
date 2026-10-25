#!/bin/bash
#SBATCH --job-name=v2_predict
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=v2_predict_%j.out
# New sweep realisation, step 5: DeepONet IL-8 predictions at every sweep point (SCALE = clip | noclip, SEED).
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${SCALE:?set SCALE}"; : "${SEED:?set SEED}"
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK="${WORK:-/gpfs/scratch1/shared/jkowalczuk/sweep_v2}"
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
mkdir -p $BASE/smores/results/v2
cd $BASE/smores/smore
python run_calibration_surrogate.py --sim-root $WORK/outputs --infer-root $WORK/infer_$SCALE \
    --weights-dir $BASE/models/v2/deeponet_$SCALE --manifest $BASE/smores/manifest.json \
    --model deeponet --run-tag run_0062 --seed $SEED --grid 50 --cytokines il8 \
    --out $BASE/smores/results/v2/calibration_surrogate_il8_${SCALE}_seed${SEED}.json
echo "DONE"
