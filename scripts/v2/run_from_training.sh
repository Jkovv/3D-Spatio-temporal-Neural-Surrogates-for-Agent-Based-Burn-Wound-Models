#!/bin/bash
# Re-run the chain of run_all.sh from the training step on: the fields and the preprocessed
# data of the current realisation stay as they are, the eight networks are tuned and trained
# again, then predictions, loop and evaluation follow. Afterwards: scripts/v2/collect.sh and
# sbatch scripts/v2/figures.slurm.
# Run on a login node:  bash scripts/v2/run_from_training.sh
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK=/gpfs/scratch1/shared/jkowalczuk/sweep_v2
export WORK

for d in prep_clip/run_0062 prep_noclip/run_0062 infer_clip infer_noclip; do
    [ -d $WORK/$d ] || { echo "missing $WORK/$d (run prep62.slurm / prep_infer.slurm first)"; exit 1; }
done
mkdir -p $BASE/models/v2 $BASE/smores/results/v2 $BASE/smores/results_v3/v2
cd $BASE

TR="--export=ALL,WORK=$WORK,OUTROOT=$BASE/models/v2"
T_DC=$(sbatch --parsable "$TR,MODEL=deeponet,SCALE=clip,CYTS=il8 il10" scripts/clipcmp_train.slurm)
T_UC=$(sbatch --parsable "$TR,MODEL=unet,SCALE=clip,CYTS=il8 il10" scripts/clipcmp_train.slurm)
T_DN=$(sbatch --parsable "$TR,MODEL=deeponet,SCALE=noclip,CYTS=il8" scripts/clipcmp_train.slurm)
T_UN=$(sbatch --parsable "$TR,MODEL=unet,SCALE=noclip,CYTS=il8" scripts/clipcmp_train.slurm)
echo "training: $T_DC $T_UC $T_DN $T_UN"

for S in clip noclip; do
    if [ "$S" = "clip" ]; then T=$T_DC; else T=$T_DN; fi
    for SEED in 42 1 100; do
        PR=$(sbatch --parsable --dependency=afterok:$T --export=ALL,WORK=$WORK,SCALE=$S,SEED=$SEED scripts/v2/predict.slurm)
        LP=$(sbatch --parsable --dependency=afterok:$PR --export=ALL,SCALE=$S,SEED=$SEED scripts/v2/loop.slurm)
        echo "$S seed $SEED: predict $PR, loop $LP"
    done
done

EV=$(sbatch --parsable --dependency=afterok:$T_DC:$T_UC:$T_DN:$T_UN --export=ALL,WORK=$WORK scripts/v2/eval.slurm)
echo "eval: $EV"
squeue -u $USER
