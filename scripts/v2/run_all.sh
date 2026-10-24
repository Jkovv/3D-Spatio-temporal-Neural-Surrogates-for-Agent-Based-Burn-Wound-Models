#!/bin/bash
# New realisation of the sweep and everything that depends on the fields, submitted as one chain:
#   1 sweep (array of 100 ABM runs)  2 run_0062 preprocessing  3 all points on the frozen scale
#   4 training (clipped: IL-8 + IL-10 for both networks; unclipped: IL-8)  5 predictions  6 loop  7 evaluation
# Run on a login node:  bash scripts/v2/run_all.sh
set -euo pipefail
BASE=/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/3D-Spatio-temporal-Neural-Surrogates-for-Agent-Based-Burn-Wound-Models
WORK=/gpfs/scratch1/shared/jkowalczuk/sweep_v2
PY=/gpfs/scratch1/shared/jkowalczuk/miniconda3/envs/cc3d/bin/python
export WORK
mkdir -p $WORK $BASE/models/v2 $BASE/smores/results/v2 $BASE/smores/results_v3/v2

cd $BASE/smores
$PY run_sweep.py --manifest manifest.json --mode slurm --cc3d-python $PY --cc3d-file $BASE/smores/combi3D.cc3d \
    --runs-dir $WORK/runs --out $WORK/outputs --work $WORK --account gisr115048 --cpus 4 --hours 4
SWEEP=$(sbatch --parsable $WORK/sweep_array.sh)
echo "sweep array: $SWEEP"

cd $BASE
P62=$(sbatch --parsable --dependency=afterok:$SWEEP scripts/v2/prep62.slurm)
PI=$(sbatch --parsable --dependency=afterok:$P62 scripts/v2/prep_infer.slurm)
echo "prep: $P62 $PI"

TR="--export=ALL,WORK=$WORK,OUTROOT=$BASE/models/v2"
T_DC=$(sbatch --parsable --dependency=afterok:$P62 "$TR,MODEL=deeponet,SCALE=clip,CYTS=il8 il10" scripts/clipcmp_train.slurm)
T_UC=$(sbatch --parsable --dependency=afterok:$P62 "$TR,MODEL=unet,SCALE=clip,CYTS=il8 il10" scripts/clipcmp_train.slurm)
T_DN=$(sbatch --parsable --dependency=afterok:$P62 "$TR,MODEL=deeponet,SCALE=noclip,CYTS=il8" scripts/clipcmp_train.slurm)
T_UN=$(sbatch --parsable --dependency=afterok:$P62 "$TR,MODEL=unet,SCALE=noclip,CYTS=il8" scripts/clipcmp_train.slurm)
echo "training: $T_DC $T_UC $T_DN $T_UN"

for S in clip noclip; do
    if [ "$S" = "clip" ]; then T=$T_DC; else T=$T_DN; fi
    for SEED in 42 1 100; do
        PR=$(sbatch --parsable --dependency=afterok:$PI:$T --export=ALL,WORK=$WORK,SCALE=$S,SEED=$SEED scripts/v2/predict.slurm)
        LP=$(sbatch --parsable --dependency=afterok:$PR --export=ALL,SCALE=$S,SEED=$SEED scripts/v2/loop.slurm)
        echo "$S seed $SEED: predict $PR, loop $LP"
    done
done

EV=$(sbatch --parsable --dependency=afterok:$T_DC:$T_UC:$T_DN:$T_UN --export=ALL,WORK=$WORK scripts/v2/eval.slurm)
echo "eval: $EV"
squeue -u $USER
