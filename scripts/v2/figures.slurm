#!/bin/bash
#SBATCH --job-name=v2_figures
#SBATCH --account=gisr115048
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=v2_figures_%j.out
# New sweep realisation, step 8: make the new benchmark the canonical one on the cluster
# (weights and preprocessed run_0062), then rebuild every figure and the panels.
# Run after `git pull` of the commit that moved the new results into models/ and smores/results*/.
set -euo pipefail
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK="${WORK:-/gpfs/scratch1/shared/jkowalczuk/sweep_v2}"
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
cd $BASE
# weights of the new benchmark next to their result files
for M in deeponet unet; do
    mkdir -p models/${M}_3d
    rm -f models/${M}_3d/*.weights.h5
    cp models/v2/${M}_clip/*.weights.h5 models/${M}_3d/
done
# preprocessed benchmark run (clipped protocol) in the canonical place
rm -rf preprocessed_3d/run_0062
mkdir -p preprocessed_3d
cp -r $WORK/prep_clip/run_0062 preprocessed_3d/
python -m py_compile scripts/figures.py scripts/fig_seeddots.py scripts/assemble_panels.py
python scripts/figures.py --models-root models --prep-root preprocessed_3d --sweep-root smores --grid 50 --out figures_3d
python scripts/fig_seeddots.py
python scripts/assemble_panels.py --in figures_3d --out figures
cp figures_3d/F6_rollout.png figures/fig8_rollout.png
if [ -n "${SKIP_STATS:-}" ]; then echo "DONE (figures only)"; exit 0; fi
# field statistics of the sweep for the data appendix (conda python, without the TensorFlow module environment)
module purge
unset PYTHONPATH
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
$PY smores/helpers/dataset_stats.py $WORK/outputs > smores/helpers/dataset_stats_report.txt
$PY smores/helpers/sparsity_both.py $WORK/outputs > smores/helpers/sparsity_report.txt
cat smores/helpers/sparsity_report.txt
ls -lt figures_3d | head -20
echo "DONE"
