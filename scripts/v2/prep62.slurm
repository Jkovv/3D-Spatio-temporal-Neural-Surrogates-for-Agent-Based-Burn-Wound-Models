#!/bin/bash
#SBATCH --job-name=v2_prep62
#SBATCH --account=gisr115048
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=v2_prep62_%j.out
# New sweep realisation, step 2: the benchmark run with its own fitted scale, clipped and unclipped.
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK="${WORK:-/gpfs/scratch1/shared/jkowalczuk/sweep_v2}"
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
cd $BASE/scripts
$PY preprocessing/preprocess_3d.py --sim-root $WORK/outputs --out-root $WORK/prep_clip --grid 50 --runs run_0062
$PY preprocessing/preprocess_3d.py --sim-root $WORK/outputs --out-root $WORK/prep_noclip --grid 50 --runs run_0062 --no-clip
$PY - <<EOF
import json
for s in ("clip", "noclip"):
    m = json.load(open("$WORK/prep_" + s + "/run_0062/50x50x50/metadata.json"))["scaling"]
    print(s, "clip percentile (il8 il1 il6 il10 tnf tgf):", m["clip_percentile"], "kurtosis:", [round(k, 1) for k in m["excess_kurtosis"]])
EOF
echo "=== DONE ==="
