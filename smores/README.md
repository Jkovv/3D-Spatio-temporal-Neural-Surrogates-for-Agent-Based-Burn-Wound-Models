# Sensitivity analysis and SMoRe ParS inference for the 3D burn ABM

This directory holds the 100-point Latin-hypercube sweep of the ABM, its replicate runs, the Sobol sensitivity analysis, and parameter inference with SMoRe ParS (Jain et al. 2022; Bergman et al. 2024), including the surrogate-in-the-loop experiment and the external stress test on measured *E. coli* growth curves. The results in the manuscript come from the scripts in `smore_strict/` and are stored in `results_v3/`.

## Layout

```
smores/
├── manifest.json               # single source of truth: 100 runs × 10 params, LHS seed 42
├── manifest_replicates.json    # the five additional realisations of every sweep point
├── setup_runs.py, run_sweep.py, param_loader.py, combi3D*.py, params_*.py ...   # ABM and sweep staging
├── smore_strict/               # inference pipeline used for the manuscript (see below)
├── helpers/                    # supporting analyses (surrogate-model selection, ridge, benchmark run, ...)
├── smore/                      # shared modules (observables, Sobol emulator) and the script that writes the loop's surrogate trajectories
├── external_data/              # place the supplementary spreadsheets of Gong & Ying (2026) here; not included
├── results/                    # inputs (replicate trajectories, Sobol indices, ridge analysis, SM selection, loop trajectories)
│                               # and the results of the previous run of the pipeline
├── results_v3/                 # results used in the manuscript, and the generated tables (results_v3/tables/)
└── sweep/outputs/run_0001 … run_0100/   # ABM output: params.json, datafiles/, LatticeData/{Cyto,Cell}Step_*.npz
```

`manifest.json` carries `param_names`, `bounds` (per parameter `low`/`high` plus a description), `baselines`, and the per-run vectors under `runs`. `params.json` in each run directory nests the vector under a `params` key and is the pairing key between θ and trajectory.

---

## The sweep

### Parameter injection

Per-run-directory staging, matching the original sweep's structure and safer than an environment variable, since it does not depend on CC3D passing the environment through to the steppable process. `run_sweep.py` stages `sweep/runs/<run_id>/Simulation/` with a full copy of the code plus a validated `params.json`; `param_loader.py` reads that local file (or `$SMORE_PARAMS` if set); CC3D writes to `sweep/outputs/<run_id>/`, outside the run directory as CC3D requires, and `params.json` is copied there. CC3D is launched with:

```
<cc3d_python> -m cc3d.run_script --input=<run>/combi3D.cc3d --output-dir=<out>
```

`install_cc3d.slurm` installs CompuCell3D on the cluster (one-time, about 1-2 h).

### Sampling and replicates

Latin hypercube via `scipy.stats.qmc.LatinHypercube` (`setup_runs.py`), deterministic under a fixed seed and independent of SALib. Every range spans ±30% around the baseline value of the inherited model; the four initial cell counts are rounded to the nearest integer, so the two end values of each count receive half the probability of an interior value. The ABM is stochastic, so every sweep point was simulated five more times (`manifest_replicates.json`), giving six realisations per point. `smore_strict/collect_replicates.py` averages each cytokine field over the whole $50^3$ domain at each of the 101 hourly frames, as written by the ABM (no noise floor, no clipping), and stores the trajectories of all realisations in `results/replicates.npz`:

```bash
python smore_strict/collect_replicates.py --sweep sweep/outputs --replicates <replicate-output-dir> \
    --out results/replicates.npz --nrep 5 --frames 101 --procs 16
```

---

## Inference pipeline (`smore_strict/`)

| Script | What it does |
|---|---|
| `sobol_repmean.py` | Gaussian-process emulator per observable on the replicate-mean volume averages, Sobol indices on a Saltelli sample of it (base sample 1024). Four observables per cytokine: final value, time-average, maximum, area under the curve |
| `sobol_diagnostics.py` | Five-fold cross-validated $R^2$ of every emulator, and the indices recomputed on 60-100 sweep points |
| `sobol_per_cytokine.py` | The indices per cytokine, as in the manuscript table |
| `../helpers/sm_selection.py` | Choice of the surrogate model (SM) per cytokine by AIC among three ODE candidates |
| `smore_pars_strict.py` | SMoRe ParS: SM fits with profile-likelihood bounds, Gaussian-process bound surfaces, leave-one-out admissible regions; `--thin k` fits every k-th frame |
| `loop_strict.py` | Surrogate predictions as data: five ways of forming the data interval from the DeepONet's IL-8 trajectories, and a matched-error control |
| `ecoli_strict.py` | The same workflow on the *E. coli* growth curves, with a positive control (`--drop-genome` leaves out one strain) and genome classification |
| `make_tables.py` | All manuscript tables of this part, and `numbers.txt`: every number quoted in the text with its source |
| `run_v3_main.slurm`, `run_v3_ecoli.slurm` | Cluster drivers of the manuscript version; adjust `BASE` and `PY` at the top |
| `run_v2_loop.slurm` | Driver of the loop, whose results are unchanged and copied into `results_v3/` |

```bash
sbatch smore_strict/run_v2_loop.slurm    # surrogate predictions as data, training seeds 1, 42, 100
sbatch smore_strict/run_v3_main.slurm    # main, every 5th frame, quadrature, top five, one SD
sbatch smore_strict/run_v3_ecoli.slurm   # real curves, positive control (five and four strains), continuous box, quadrature
python smore_strict/make_tables.py --results results_v3 --sobol results/sobol_repmean.json --ecoli-npz results_v3/ecoli_arrays.npz
```

Each driver can first be run on a few folds as a check: `sbatch --export=ALL,FOLDS=3 smore_strict/run_v3_main.slurm` (output in `results_v3_test/`). The loop needs the volume-averaged DeepONet and ABM trajectories at all sweep points, `results/calibration_surrogate_il8*_trajectories.npz`, written by `smore/run_calibration_surrogate.py`.

---

## Method notes

**Observable.** Volume averages of the cytokine fields over the $50^3$ domain, taken from the raw fields, before any surrogate preprocessing. The six realisations of a sweep point give a mean trajectory and its standard error per cytokine. Time is scaled to $[0,1]$ for the SM fits.

**Surrogate models.** For each cytokine the candidate with the lowest AIC in the largest number of runs, which also has the lowest summed AIC: IL-8 production from a decaying source (M3, four parameters; 99 of 100 runs); IL-6, IL-10, TNF-α production with decay (M1, three; 91, 91, 72 runs); TGF-β logistic (M2, three; 84 runs). IL-1β is excluded because no candidate describes its separate secretion events. In the inherited model IL-10 is secreted by the same M1 cells as IL-6 with the same decay rate; since diffusion does not change the total amount in the domain, its volume average is the IL-6 average scaled by `km2il10`/`km1il6`, although the diffusion coefficients differ. The two SMs share their shape and differ only in amplitude, so IL-6 and IL-10 do not provide independent shape information.

**Fits and bounds.** For numerical conditioning each trajectory and its errors are divided by the trajectory's maximum before fitting; the amplitude parameters ($c_0$, $p$, $A$) and their bounds are multiplied back by that maximum, so the SM parameters are in physical units and keep the amplitude. Weighted least squares with the replicate standard error as weight, as in Jain et al.; standard errors below 5% of their median are raised to it. Volume-averaged IL-8 is deterministic to about $10^{-6}$ of its maximum, far below the error of the SM itself, so for a cytokine whose median standard error is below $10^{-4}$ of the maximum (IL-8 at every point) the RMS residual of an unweighted fit is used instead. 95% bounds per SM parameter from the profile likelihood ($\Delta\chi^2 = 3.84$). The fit treats the frames as independent, which makes the intervals too narrow; `--thin 5` fits every fifth frame to assess this. With `--sigma-mode quadrature` the SM misfit is added to the replicate error for every cytokine.

**Unidentified SM parameters.** A parameter whose best fit lies at an upper limit of its allowed range (or at a lower limit other than zero), or whose profile interval spans the whole range, is not determined by that trajectory. Such a point is left out of the bound surfaces of that parameter only. On the ABM this concerns the initial value of IL-6 and IL-10 at all points (those surfaces are dropped) and two TNF-α parameters at 39 and 4 points.

**Bound surfaces and region.** Each bound is interpolated over the ten ABM parameters with a Gaussian process (constant × Matérn-5/2 with ARD, plus white noise; positive parameters on a log scale), fitted without the held-out point. An ABM parameter vector is admissible if, for every SM parameter, its predicted interval, widened by two predictive standard deviations, overlaps the data interval. The region is sampled by rejection on a scrambled Sobol sequence (up to $4\,194\,304$ points, until 300 are admissible), so the sample is uniform and its share is the region's volume fraction. Smaller regions are completed by hit-and-run chains, extended until the split $\hat R$ of the second half is at most 1.1 (up to 16 000 steps per walker); folds whose chain does not converge are left out of the sample-dependent statistics and keep only the admissibility of the true vector. The four initial cell counts are treated as continuous in the box, so the region is a continuous relaxation and its volume fraction refers to the relaxed box. Annealing finds a start when no point is admissible. Bergman et al.'s variant without interpolation accepts no sweep point at any held-out target with 100 points in ten dimensions.

**Evaluation.** Leave-one-out over the sweep points. Reported per fold: whether the true vector itself is admissible (empirical joint containment; the region intersects marginal constraints and is not a joint 95% confidence region); per parameter the marginal containment of the central 95% of the region, its width (SD of the region / SD of a uniform over the range), and the $R^2$ of its median; a model-free reference, a Gaussian-process regression from the fitted SM parameters straight to each ABM parameter; and whether the held-out bounds lie within the band of the interpolated surfaces. Per-fold medians and quantiles of the region are stored in the result files.

**Surrogate predictions as data.** The bound surfaces always come from the ABM; only the held-out point's data interval changes. Arms: ABM output; surrogate without allowance for its error; with its validation error on the training run; with its per-frame error over the other sweep points, as if independent between frames; with the error propagated in SM-parameter space (an envelope of the differences between the SM parameters fitted to surrogate and ABM trajectories over the other 98 points; the 0.625% and 99.375% quantiles reduce to their minimum and maximum); and a control, the held-out ABM trajectory plus the surrogate's error trajectory from another, random point.

**External data.** 5 strains × 29 media × 6 replicate growth curves (Gong & Ying 2026). The readings of the 4.63 Mb strain end at 24 h, those of the other strains at 48.5 h. The media vary one component at a time, and K⁺/PO₄³⁻ and NH₄⁺/SO₄²⁻ vary together because they come from the same salts. The region is therefore evaluated on the 145 conditions of the design rather than on a continuous box, which consists mostly of media that do not exist; the genome is also classified by the genome with the most admissible conditions. The positive control keeps the design, the replicate residuals and the missing readings, and generates curves in which genome size sets the plateau, K⁺ the rate and NH₄⁺ the midpoint, each over the 5-95% range of the real fits. With all five strains the plateau of the 4.63 Mb strain is not identified within its window, so the control is also run without that strain.

**Endothelium is frozen.** `init_ec` affects IL-8 only through the number of constitutive sources, and the IL-8 equation is linear with fixed sources, so volume-averaged IL-8 depends on `init_ec` × `keil8`. Over the 100 simulated points the replicate-mean final IL-8 is linear in the product with 1 - R² = 1.9e-9 and correlates with each factor at 0.70 and 0.69; a GP emulator gives the same picture (`results/ridge_init_ec_keil8_repmean_*.json`). The small deviation from an exact product comes from the neutrophil terms, so this is a practical product non-identifiability under the volume-averaged observable. On fields clipped at the surrogate's training scale the dependence on the product breaks, and the two parameters become separately recoverable; this separability is induced by preprocessing.

---

## Results (`results_v3/`, tables in `results_v3/tables/`)

| Analysis | Outcome |
|---|---|
| Sobol, per cytokine | Each secretion rate dominates its own cytokine (0.42-0.56); `keil8` and `init_ec` 0.507 each on IL-8; `sigmoidb` acts on five cytokines |
| Sobol, 24 observables | `sigmoidb` 0.575 first, then `km1il6`, `km2il10`, `init_ec`, `keil8`, `km2tgf` (0.069-0.094) without a stable order; the other four ≤ 0.003 |
| SMoRe ParS, 10 parameters | `sigmoidb` R² = 0.88 (width 0.40); `km1il6` 0.74, `km2tgf` 0.72, `km2il10` 0.69; `init_ec`, `keil8` ≈ 0.50 each, constrained only through their product (width 0.058 along the sum, 0.971 along the difference); `lnril8`, `init_n`, `init_m`, `init_f` not recovered. 15 folds needed chains, all with R-hat ≤ 1.1 (maximum 1.095) |
| Containment | Marginal containment 0.94-0.98 (`init_f` 0.83); true vector admissible in 84% of folds, held-out bounds within the 2 SD band in a median of 92% |
| Integer cell counts | The region is sampled as a continuous relaxation of the four integer counts. Rounded to integer counts (from the per-fold region quantiles stored in `results_v3/smore_main.json`), the containment of `init_f` rises from 0.83 to 1.00 (its true value is on a range end in 16 folds) and that of the other counts changes by at most 0.02; the region medians do not change |
| Error model | Every 5th frame: true vector admissible in 97% of folds, region 2.4 times larger, secretion-rate R² 0.59-0.67. SM misfit in quadrature: 99%, region 3.3 times larger. Top five: same ordering and ridge. One SD: 35% (only containment reported) |
| Surrogate predictions as data | Trajectory R² 0.93-0.98; error at unseen points 3.7-5.6% against 0.8-0.9% on the training run's validation frames. Without allowance the true vector is admissible at ≤ 4% of points; with the error propagated in SM space at 98-100%, but the region covers 50-77% of the box instead of 2%, and `keil8` R² falls from 0.87 to 0.13-0.28. The matched-error control loses almost as much |
| *E. coli*, real curves | Logistic SM median R² 0.983 on 126 growing conditions; median 13 of 145 conditions admissible per curve; genome R² 0.69, true genome among the admissible conditions for 92% of curves, most-admissible genome correct for 60% (chance 20%); true condition admissible in 81% (92% with misfit). On the continuous box the region covers 96.5% of it |
| *E. coli*, positive control | Five strains: rate and midpoint inputs recovered (R² 0.66, 0.76); genome median R² 0.00 but most-admissible genome correct for 75%. Four strains (4.63 Mb strain left out): genome identified in every fold (R² 1.00), rate and midpoint inputs 0.94 and 1.00, inactive inputs unconstrained, true condition admissible in 100% |

The surrogate row is the one worth internalising before reusing this code: a surrogate that reproduces trajectories closely can still be unusable as data for inference unless its error away from the training data is measured and propagated, and even then little parameter information may remain.
