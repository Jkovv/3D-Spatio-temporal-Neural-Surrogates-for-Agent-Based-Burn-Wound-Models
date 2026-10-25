# 3D Spatio-temporal Neural Surrogates for Agent-Based Burn-Wound Models

Code and results for *Forward fidelity does not guarantee parameter recoverability in a three-dimensional agent-based burn-wound model*: volumetric neural field surrogates for a three-dimensional agent-based model (ABM) of the post-burn immune response, a global sensitivity analysis of the ABM, and parameter inference with SMoRe ParS, including a test of whether surrogate predictions can serve as data for that inference and an external stress test on measured *E. coli* growth curves.

All ABM results come from one 100-point Latin-hypercube design (`smores/manifest.json`). The analyses that work on volume-averaged trajectories (Sobol, SMoRe ParS, the ridge analysis) use six stochastic realisations of every point, stored as trajectories in `smores/results/replicates.npz`. The analyses that need the full cytokine fields (surrogate training, rollouts, the clipping comparison, surrogate predictions as data) use one realisation of the same 100 points, simulated in one batch (`scripts/v2/run_all.sh`); the ABM has no fixed RNG seed, so this is a separate draw. Between realisations volume-averaged IL-8 differs by less than 0.1%, the other cytokines by 10-30%. Surrogate hyperparameters were tuned with Optuna (40 trials; 60 tuning epochs for DeepONet, 40 for the U-Net) on seed 42 and reused for seeds 1 and 100. All Gaussian-process fits and the region search use fixed random seeds.

---
## Headline results

| | |
|---|---|
| One-step accuracy, dense cytokine (IL-8) | DeepONet R² = 1.000, 3D U-Net 0.996, against the training target, which is clipped at the 99th percentile of IL-8 |
| One-step accuracy, sparse cytokine (IL-10, not clipped) | DeepONet 0.903 (0.84-0.94 across seeds) in the early and 0.51 (-0.31-0.95) in the late test window; U-Net 0.721 (0.67-0.81) and 0.607 |
| Trivial baselines | Repeating the previous frame scores higher than both networks: R² = 0.9999 on IL-8 and 0.967 / 0.958 on IL-10 (early / late window); linear extrapolation reaches 0.934 / 0.918 on IL-10. In this teacher-forced one-step setting the field metrics mostly reflect how slowly the fields change between hours |
| Clipping | Scored against the full IL-8 field, networks trained on the clipped field reach R² = 0.71, the ceiling for any predictor of the clipped field (the voxels above the clip value are 1.7% of the field and carry 29% of its variance). Trained without clipping, the U-Net reaches 0.95 and DeepONet 0.52-0.85, still below persistence (`models/clip_comparison/`) |
| Autoregressive rollouts | From t = 82 h, with the network's own channel fed back, no network is stable across seeds: on IL-8 one DeepONet seed collapses (late R² -0.01) while the U-Net holds 0.93-0.96; on IL-10 one DeepONet seed beats persistence and two collapse (`models/rollout_3d/`) |
| Mid-plane accuracy | On IL-10 the xy plane is the weakest for DeepONet (0.82 against 0.90-0.91) and the yz plane for the U-Net (0.69 against 0.78-0.94) |
| Cost of a field read-out | One ABM trajectory (5121 s) takes 2700-3500 times longer than a U-Net read-out and 125 times longer than a DeepONet read-out. The surrogates need the ABM's cell masks and the two previous frames, so this is the cost of reading out a field along a simulated trajectory, not a saved simulation |
| Sensitivity | Per cytokine, each secretion rate dominates its own signal (S_T 0.42-0.56) and `keil8` and `init_ec` each get 0.507 on IL-8. Over all 24 observables `sigmoidb` leads (S_T = 0.575) because it acts on five cytokines |
| Recovery (SMoRe ParS, leave-one-out, 10 parameters) | `sigmoidb` R² = 0.88; `km1il6`, `km2tgf`, `km2il10` 0.69-0.74; `init_ec` and `keil8` only through their product (region width 0.058 along their sum, 0.971 along their difference); `lnril8` and the other initial counts not recovered. Unchanged on nested subsamples of 60-90 sweep points (`results_v3/tables/tab_learning_curve.tex`) |
| Sensitivity ≠ identifiability | `init_ec` and `keil8` receive identical Sobol indices, yet the data constrain only their product: endothelial cells are frozen, so volume-averaged IL-8 is linear in sources × secretion rate (1 - R² = 1.9e-9; log-linear exponents 1.0000 and 1.0001). The two parameters are uncorrelated in the design (r = -0.03) |
| Containment | Marginal containment 0.94-0.98 for most parameters (`init_f` 0.83; rounded to integer counts 1.00). The true parameter vector is admissible in 84% of folds with replicate error alone, 97% when every fifth frame is fitted, 99% with the surrogate-model misfit included, at the cost of wider regions |
| Surrogate predictions as data | DeepONet IL-8 trajectories reach R² 0.96-0.99, but their error at unseen sweep points (2.5-4.1%) is 3-6× the validation error on the training run (0.4-0.8%). Without allowance for it, the inferred regions contain the true parameters at ≤ 1% of points; with the error propagated in surrogate-model parameter space at 99-100%, but they then cover 24-73% of the parameter space instead of 2%, and `keil8` R² drops from 0.89 to 0.09-0.39. A surrogate trained without clipping has an out-of-sample error of 27-147% and is unusable as data (`results_v3/tables/tab_loop_unclipped.tex`) |
| External stress test (*E. coli*) | Logistic surrogate model, median R² = 0.983 on 126 growing conditions. Per curve a median of 13 of the 145 design conditions remain admissible; the genome with the most admissible conditions is the true one for 60% of curves (chance 20%). True condition admissible in 81% (92% with misfit). A positive control recovers the active inputs once the 4.63 Mb strain, whose readings end at 24 h, is left out |

---

## Layout

```
.
├── models/      # result files of the surrogates (weights are not in the repository)
│   ├── deeponet_3d/, unet_3d/        # benchmark: clipped protocol, IL-8 and IL-10, seeds 1/42/100
│   ├── clip_comparison/              # the same networks trained without clipping, scored against the full field
│   ├── rollout_3d/                   # autoregressive rollouts
│   └── persistence_baseline_run_0062_50.json, forward_numbers.json
├── scripts/     # preprocessing, training, evaluation, figures; scripts/v2/ is the chain that produced the field results
├── figures/     # the manuscript figures (PNG)
├── figures_3d/  # individual panels the manuscript figures are assembled from
└── smores/      # sweep, sensitivity analysis and SMoRe ParS (see smores/README.md)
```

**`scripts/`** holds the preprocessing from the 2D benchmark (kurtosis-adaptive percentile clipping, two-frame look-back, chronological 70/10/19 split), adapted to 3D by removing a numerical-diffusion noise floor and computing the kurtosis and clip percentile over the voxels above it (`preprocessing/preprocess_3d.py`; `--no-clip` keeps a linear scale). Of the two benchmark cytokines only IL-8 is clipped (99th percentile); IL-10 has an excess kurtosis below 20 and is not. Training scripts: `deeponet/train_deeponet_3d.py`, `unet/train_unet_3d.py` (metrics: global R², masked RMSE, Dice, volumetric SSIM, Fisher-z-pooled spatial correlation, mid-plane R²). Evaluation: `persistence_baseline.py` (persistence and linear extrapolation with the same metric code), `rollout_3d.py` and `rollout_summary.py` (autoregressive rollouts, table and figure), `clip_comparison.py` and `clip_ceiling.py` (clipped vs unclipped normalisation against the full field), `forward_numbers.py` (the numbers of the forward tables from the result files), `diag_late_window.py` (per-frame R² over active and empty voxels, background offset and sign of the predictions; `run_diag.slurm`). Figures: `figures.py`, `fig_seeddots.py`, `assemble_panels.py`.

**`scripts/v2/`** is the chain that produced every field-based result, submitted as one set of dependent Slurm jobs by `run_all.sh`: the 100 ABM runs, preprocessing of the benchmark run and of every sweep point on its frozen scale (clipped and unclipped), training of both networks on both normalisations, DeepONet predictions at every sweep point, the surrogate-in-the-loop inference, and the evaluation (`eval.slurm`). `run_from_training.sh` repeats the chain from the training step on the existing fields, `collect.sh` copies the results into the directories below and rebuilds the tables, and `figures.slurm` rebuilds the figures and the field statistics.

**`figures/`**: one PNG per manuscript figure. File numbers follow the order in which the figures were made, not their numbering in the paper.

| File | Content |
|---|---|
| `fig1_data.png` | Cytokine trajectories and the dense/sparse contrast (slices at t = 88 h) |
| `fig2_architectures.png` | The two benchmarked architectures |
| `fig3_accuracy.png` | One-step volumetric accuracy and metrics, per seed |
| `fig4_midplane.png` | Mid-plane reconstruction at t = 88 h and per-plane R² |
| `fig5_cost.png` | Inference time relative to one ABM trajectory |
| `fig6_sobol.png` | Sobol total-order indices, ranked |
| `fig7_smore.png` | SMoRe ParS on the ABM: per-cytokine sensitivity against recovery, and region widths |
| `fig8_rollout.png` | Autoregressive rollouts, R² per step ahead |

**`smores/`** is the sweep and the inference pipeline; `smores/README.md` describes it. The manuscript tables are generated from the result files: `smores/results_v3/tables/` (`make_tables.py`), `models/rollout_3d/tab_rollout.tex` (`rollout_summary.py`) and `models/clip_comparison/tab_clip_comparison.tex` (`clip_comparison.py --table-only`).

---

## Reproducing

The cluster drivers contain the Snellius paths of the authors' environment; adjust `BASE`, `WORK` and `PY` at the top of each. Raw cytokine fields are not in the repository (about 20 GB for the 100 runs).

```bash
# field-based results: ABM runs, preprocessing, training, predictions, loop, evaluation (one dependent chain)
bash scripts/v2/run_all.sh
bash scripts/v2/run_from_training.sh      # or: retune and retrain the networks on the existing fields
bash scripts/v2/collect.sh                # afterwards: result files into models/ and smores/, tables
sbatch scripts/v2/figures.slurm           # figures, panels, field statistics

# volume-average-based results (replicate trajectories in smores/results/replicates.npz)
cd smores
sbatch smore_strict/run_v3_main.slurm           # SMoRe ParS: main, every 5th frame, quadrature, top five, one SD
sbatch smore_strict/run_v3_learning_curve.slurm # the same on 60-90 sweep points
sbatch smore_strict/run_v3_ecoli.slurm          # external test (needs the spreadsheets of Gong & Ying in smores/external_data/)
python smore_strict/design_check.py --manifest manifest.json --replicates-npz results/replicates.npz --out results/design_check.json
python smore_strict/make_tables.py --results results_v3 --sobol results/sobol_repmean.json --ecoli-npz results_v3/ecoli_arrays.npz

# tables and chart figures from the result files (no cluster needed)
cd ..
python scripts/forward_numbers.py
python scripts/rollout_summary.py
python scripts/clip_comparison.py --work . --models models/clip_comparison --table-only
python scripts/figures.py --figs B1 B4 B3 E4 K1 K3
```

Surrogate results were produced on single NVIDIA A100 GPUs on Snellius; the ABM runs, the Sobol analysis and SMoRe ParS on its CPU nodes.
