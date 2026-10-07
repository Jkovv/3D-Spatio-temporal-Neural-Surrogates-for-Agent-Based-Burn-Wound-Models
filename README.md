# 3D Spatio-temporal Neural Surrogates for Agent-Based Burn-Wound Models

Code and results for *Forward fidelity does not guarantee parameter recoverability in a three-dimensional agent-based burn-wound model*: volumetric neural field surrogates for a three-dimensional agent-based model (ABM) of the post-burn immune response, a global sensitivity analysis of the ABM, and parameter inference with SMoRe ParS, including a test of whether surrogate predictions can serve as data for that inference and an external stress test on measured *E. coli* growth curves.

All ABM results come from a single 100-point Latin-hypercube sweep, with six stochastic realisations per point. The ABM has no fixed RNG seed, so the sweep, surrogates, sensitivity analysis, inference and figures were all recomputed together on this sweep. Surrogate hyperparameters were tuned with Optuna on seed 42 and reused unchanged for seeds 1 and 100. All Gaussian-process fits and the region search use fixed random seeds, so the sensitivity and inference results are reproducible.

---
## Headline results

| | |
|---|---|
| Dense cytokine (IL-8) | DeepONet R² = 0.999, 3D U-Net R² = 0.987 |
| Sparse cytokine (IL-10) | DeepONet R² = 0.929 (0.89-0.96 across seeds; 0.956 in the late test window); U-Net 0.742 (0.62-0.82; 0.703 late). Both windows are one-step predictions from the true preceding ABM frames |
| Trivial baselines | Repeating the previous frame scores higher than both networks: R² = 0.9999 on IL-8 and 0.963 / 0.975 on IL-10 (early / late window); linear extrapolation from the last two frames reaches 0.926 / 0.952 on IL-10, about as good as DeepONet. In this teacher-forced one-step setting the field metrics mostly reflect how slowly the fields change between hours |
| Mid-plane accuracy | On IL-10 xy is the weakest plane (DeepONet 0.80, U-Net 0.73) and yz the strongest (0.97, 0.95), for both models |
| Cost of a field read-out | One ABM trajectory takes about 3100-3400 times longer than a U-Net read-out and 120 times longer than a DeepONet read-out. The surrogates need the ABM's cell masks and the cytokine fields of the two previous frames, so this is the cost of reading out a field along a simulated trajectory, not a saved simulation |
| Sensitivity | Per cytokine, each secretion rate dominates its own signal (S_T 0.42-0.56) and `keil8` and `init_ec` each get 0.507 on IL-8. Over all 24 observables `sigmoidb` leads (S_T = 0.575) because it acts on five cytokines |
| Recovery (SMoRe ParS, leave-one-out, 10 parameters) | `sigmoidb` R² = 0.88; `km1il6`, `km2tgf`, `km2il10` 0.69-0.74; `init_ec` and `keil8` only through their product (region width 0.058 along their sum, 0.971 along their difference); `lnril8` and the other initial counts not recovered. All hit-and-run chains reach split R-hat ≤ 1.1 |
| Sensitivity ≠ identifiability | `init_ec` and `keil8` receive identical Sobol indices, yet the data constrain only their product: endothelial cells are frozen, so volume-averaged IL-8 is linear in sources × secretion rate (1 - R² = 1.9e-9 over the 100 simulated points) |
| Containment | Marginal containment 0.94-0.98 for most parameters (`init_f` 0.83; the region is a continuous relaxation of the integer cell counts, and rounded to integers its containment is 1.00). The true parameter vector is admissible in 84% of folds with replicate error alone, 97% when every fifth frame is fitted (less temporal correlation), 99% with the surrogate-model misfit included, at the cost of wider regions |
| Surrogate predictions as data | DeepONet IL-8 trajectories reach R² 0.93-0.98, but their error at unseen sweep points (3.7-5.6%) is 4-6× the validation error on the training run (0.8-0.9%). Without allowance for it, the inferred regions contain the true parameters at ≤ 4% of points; with the error propagated in surrogate-model parameter space at 98-100%, but they then cover 50-77% of the parameter space instead of 2%, and `keil8` R² drops from 0.87 to 0.13-0.28 |
| External stress test (*E. coli*) | Logistic surrogate model, median R² = 0.983 on 126 growing conditions. Per curve a median of 13 of the 145 design conditions remain admissible; the genome with the most admissible conditions is the true one for 60% of curves (chance 20%). True condition admissible in 81% (92% with misfit), the same pattern as on the ABM. A positive control recovers the inputs acting on the curves once the 4.63 Mb strain, whose readings end at 24 h, is left out (genome identified in every fold) |

---

## Layout

```
.
├── models/      # DeepONet and U-Net: result files (trained weights are in the Zenodo archive)
├── scripts/     # preprocessing, training, evaluation and figure scripts
├── figures/     # the manuscript figures (PNG)
├── figures_3d/  # individual panels the manuscript figures are assembled from
├── smores/      # sweep, sensitivity analysis and SMoRe ParS - see smores/README.md
├── .gitattributes
└── .gitignore
```

**`models/`** holds the two architectures carried forward from the 2D benchmark (https://zenodo.org/records/20465819).

**`scripts/`** holds the preprocessing from the 2D benchmark - the kurtosis-adaptive percentile-clipping normalisation, the two-frame look-back, and the chronological 70/10/19 split - adapted to 3D by removing a numerical-diffusion noise floor and computing the kurtosis and clip percentile over the voxels above it. It also holds the evaluation metrics (global R², masked RMSE, Dice, volumetric SSIM, Fisher-z-pooled spatial correlation) and the orthogonal mid-plane metrics added for the 3D setting (`midplane_r2.py`). All accuracy metrics are single-step: each frame is predicted from the ABM's own two preceding frames, in an early (82-91 h) and a late (92-100 h) test window. `persistence_baseline.py` (`run_persistence_baseline.slurm` in the repository root) scores two trivial predictors with the same metric code: persistence and linear extrapolation from the true preceding frames; the result is `models/persistence_baseline_run_0062_50.json`. `figures.py`, `fig_seeddots.py` and `assemble_panels.py` rebuild all figures (`rebuild_figures.slurm`); Fig. 7 is read from `smores/results_v3/`.

**`figures/`** holds one PNG per manuscript figure. File numbers follow the order in which the figures were made, not their numbering in the paper; `fig3` and `fig7` are Figures 1 and 2 of the main text, the others appear in the appendices:

| File | Content |
|---|---|
| `fig1_data.png` | Cytokine trajectories and the dense/sparse contrast (slices at t = 88 h) - Appendix A |
| `fig2_architectures.png` | The two benchmarked architectures - Appendix B |
| `fig3_accuracy.png` | Volumetric accuracy and metrics, per seed - Figure 1 |
| `fig4_midplane.png` | Mid-plane reconstruction at t = 88 h and per-plane R² - Appendix C |
| `fig5_cost.png` | Inference time relative to one ABM trajectory - Appendix C |
| `fig6_sobol.png` | Sobol total-order indices, ranked - Appendix D |
| `fig7_smore.png` | SMoRe ParS on the ABM: per-cytokine sensitivity against recovery, and region widths - Figure 2 |

**`smores/`** is the inference pipeline:

| Path | Content |
|---|---|
| `smore_strict/smore_pars_strict.py` | SMoRe ParS (Jain et al. 2022; Bergman et al. 2024): surrogate-model fits with profile-likelihood bounds, Gaussian-process bound surfaces, leave-one-out admissible regions, hit-and-run chains run to split R-hat ≤ 1.1 |
| `smore_strict/loop_strict.py` | Surrogate predictions as data, with five ways of accounting for the surrogate's error and a matched-error control |
| `smore_strict/ecoli_strict.py` | The same workflow on the *E. coli* growth curves of Gong & Ying (2026), with a positive control and genome classification |
| `smore_strict/make_tables.py` | Builds the manuscript tables and `numbers.txt` (every number quoted in the text, with its source) from the result files |
| `smore_strict/run_v3_*.slurm` | Cluster drivers of the manuscript version |
| `results_v3/` | Results used in the manuscript, and the generated tables in `results_v3/tables/` |
| `results/` | Inputs of the analyses (replicate trajectories, Sobol indices and their diagnostics, the `init_ec` × `keil8` ridge analysis, the surrogate-model selection, the volume-averaged DeepONet and ABM trajectories used by the loop), and the results of the previous run of the pipeline |

---

## Reproducing the inference part

The cluster drivers contain the Snellius paths of the authors' environment; adjust `BASE` and `PY` at the top of each. The *E. coli* analysis expects the supplementary spreadsheets of Gong & Ying (2026) in `smores/external_data/`; they are not included here and can be obtained from the source publication.

```bash
cd smores
sbatch smore_strict/run_v2_loop.slurm    # surrogate predictions as data, three training seeds (writes results/loop_seed*.json)
sbatch smore_strict/run_v3_main.slurm    # SMoRe ParS on the ABM: main, every 5th frame, quadrature, top five, one SD
sbatch smore_strict/run_v3_ecoli.slurm   # external test, positive control with five and four strains

# after the jobs have finished
python smore_strict/make_tables.py --results results_v3 --sobol results/sobol_repmean.json --ecoli-npz results_v3/ecoli_arrays.npz

# figures and the one-step baselines
cd ..
sbatch scripts/rebuild_figures.slurm
sbatch run_persistence_baseline.slurm
```

`run_v3_main.slurm` copies the loop results from `results/` into `results_v3/` at the end. Each driver can first be run on a few leave-one-out folds as a check, e.g. `sbatch --export=ALL,FOLDS=3 smore_strict/run_v3_main.slurm`; test output goes to `results_v3_test/`.

Surrogate results were produced on a single NVIDIA A100 on Snellius.
