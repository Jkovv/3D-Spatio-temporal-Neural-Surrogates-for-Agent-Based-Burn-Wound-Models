#!/bin/bash
#SBATCH --job-name=clipcmp_prep
#SBATCH --account=gisr115048
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=clipcmp_prep_%j.out
# Clipped vs unclipped normalisation, step 1: preprocess a second realisation of run_0062
# (raw fields in $RAW) twice, with the kurtosis-adaptive clip and with --no-clip.
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK=/gpfs/scratch1/shared/jkowalczuk/clipcmp
RAW=$WORK/raw
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
cd $BASE/scripts
$PY preprocessing/preprocess_3d.py --sim-root $RAW --out-root $WORK/prep_clip --grid 50 --runs run_0062
$PY preprocessing/preprocess_3d.py --sim-root $RAW --out-root $WORK/prep_noclip --grid 50 --runs run_0062 --no-clip
$PY - <<EOF
import json
for s in ("clip", "noclip"):
    m = json.load(open("$WORK/prep_" + s + "/run_0062/50x50x50/metadata.json"))["scaling"]
    print(s, "clip percentile per cytokine (il8 il1 il6 il10 tnf tgf):", m["clip_percentile"])
    print(s, "excess kurtosis:", [round(k, 1) for k in m["excess_kurtosis"]])
EOF
echo "DONE"
