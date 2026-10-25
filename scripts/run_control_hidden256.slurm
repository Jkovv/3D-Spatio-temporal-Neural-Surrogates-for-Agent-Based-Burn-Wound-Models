#!/bin/bash
#SBATCH --job-name=ctrl_h256
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=ctrl_h256_%j.out
# Sensitivity check for DeepONet IL-10: seed 100 retrained with the trunk width of the previous
# Optuna choice (hidden = 256 instead of 128), everything else unchanged; seed 1 as a second check.
# Seed 42 is left out because the training script re-runs Optuna for it. Separate output directory.
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
OUT=$BASE/models/control_hidden256
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
mkdir -p $OUT
python - <<PY
import json
r = json.load(open("$BASE/models/deeponet_3d/res_il10_run_0062_50_42.json"))
r["best_params"]["hidden"] = 256
json.dump(r, open("$OUT/res_il10_run_0062_50_42.json", "w"), indent=4)
print("hyperparameters for the control:", r["best_params"])
PY
cd $BASE/scripts
for SEED in 100 1; do
    python deeponet/train_deeponet_3d.py --cytokine il10 --seed $SEED --run run_0062 --grid 50 --data $BASE/preprocessed_3d --out $OUT
done
python - <<PY
import json
for s in (100, 1):
    r = json.load(open("$OUT/res_il10_run_0062_50_%d.json" % s))["results"]
    print("hidden=256 seed %d: R2 early %.3f late %.3f" % (s, r["Near_Horizon_t82_t91"]["Global_R2"], r["Far_Horizon_t92_t100"]["Global_R2"]))
PY
echo "DONE"
