# SMoRe ParS calibration pipeline for the 3D burn ABM
## Layout

```
smores/
├── manifest.json              # single source of truth: 100 runs × 10 params, LHS seed 42
├── smore/                     # calibration package
│   ├── observables.py         # (θ_ABM, cell-sampled mean-concentration trajectory) per run
│   ├── sensitivity.py         # GP emulator -> Sobol ranking
│   ├── smore_pars.py          # logistic surface fit (diagnostic), θ_ABM -> observables map, leave-one-out recovery
│   ├── run_calibration.py     # orchestrates sweep -> Sobol -> SMoRe ParS
│   ├── run_calibration_surrogate.py   # same, with surrogate-predicted observables
│   ├── compare_observables.py
│   └── spatial_observables.py
├── helpers/                   # helper scripts for the experiments reported in the manuscript
│   ...
└── sweep/
    └── outputs/run_0001 … run_0100/
        ├── params.json                        # θ for this run (ground truth)
        ├── datafiles/mean_concentration.txt   # 101 rows: MCS + 6 means + 6 SDs, sampled at cell centres
        ├── datafiles/cellcount.txt
        └── LatticeData/{Cyto,Cell}Step_*.npz  # 50³ fields, 6 cytokines, 101 steps
      ...
```

`manifest.json` carries `param_names`, `bounds` (per parameter `low`/`high` plus a description), `baselines`, and the per-run vectors under `runs`. `params.json` in each run directory nests the vector under a `params` key and is the pairing key between θ and trajectory.

---

## Parameter injection

Per-run-directory staging, matching the original sweep's structure and safer than an environment variable, since it does not depend on CC3D passing the environment through to the steppable process.

`run_sweep.py` stages `sweep/runs/<run_id>/Simulation/` with a full copy of the code plus a validated `params.json`; `param_loader.py` reads that local file (or `$SMORE_PARAMS` if set); CC3D writes to `sweep/outputs/<run_id>/`, outside the run directory as CC3D requires, and `params.json` is copied there.

CC3D is launched with:

```
<cc3d_python> -m cc3d.run_script --input=<run>/combi3D.cc3d --output-dir=<out>
```

---

## Running it

```bash
# local sanity check, no CC3D required
python verify.py                      # must print ALL CHECKS PASSED

# one-time install (~1-2 h)
sbatch install_cc3d.slurm && tail -f install_<jobid>.out

# single test run - confirms CC3D produces mean_concentration.txt
sbatch test_run.slurm && tail -f testrun_<jobid>.out

# full sweep as a SLURM array (staged by test_run.slurm)
sbatch sweep/sweep_array.sh
```

Once the runs finish:

```bash
# sensitivity first, then SMoRe ParS on the top-k
python smore/run_calibration.py --sim-root sweep/outputs \
    --manifest manifest.json --top-k 5 --out calibration_results.json

# emulator audit - appendix table on how far the Sobol indices can be trusted
python helpers/emulator_cv.py --sim-root sweep/outputs --manifest manifest.json \
    --out-tex results/appendix_emulator.tex --out-csv results/emulator_cv.csv

# Sobol ranking on all 24 observables vs. the well-emulated subset
python helpers/compare_sobol.py --sim-root sweep/outputs --manifest manifest.json \
    --cv-csv results/emulator_cv.csv --threshold 0.5 \
    --out-tex results/appendix_sobol_filtered.tex

# response surface of the init_ec x keil8 ridge (final IL-8, raw product)
python helpers/ridge_raw.py

# diagnostic: does the saturating-logistic surface actually fit the ABM trajectories?
python helpers/check_surface_fit.py <repo-root>
```

`emulator_cv.py` and `compare_sobol.py` import `smore/observables.py` and `smore/sensitivity.py` rather than reimplementing anything, so the emulator they score is the emulator the indices stand on. Run `emulator_cv.py` before `compare_sobol.py` - the second consumes the first's CSV.

Note that `compare_sobol.py` takes `--n-saltelli` (default 1024). Pass the same base sample the production `sensitivity.py` run used, otherwise its "All" column will not reproduce the main Sobol table.

---

## Method notes

**Sampling.** Latin hypercube via `scipy.stats.qmc.LatinHypercube`, deterministic under a fixed seed and independent of SALib, so the Sobol step shares no RNG state with sweep generation.

**Sensitivity.** A Gaussian process is fitted per observable on the 100 real runs and Sobol indices are computed on a dense Saltelli sample of that emulator (base sample 1024); a direct Saltelli design on the ABM would need thousands of 50³ trajectories. `sensitivity._fit_gp` standardises θ and y internally and returns `(predict, gp)`. All GPs (Sobol emulator and recovery map) use `random_state=0`, so re-runs reproduce the numbers.

**Emulator quality is not uniform, and this bounds the ranking.** Cross-validating the 24 emulators (`helpers/emulator_cv.py`, pooled out-of-fold R², 5-fold, refitted per fold) gives a mean of 0.597 with a range from -0.516 to +0.990. The variation is structured along two axes at once. By cytokine: IL-8 reaches 0.982 while IL-1β reaches 0.387, the same dense-versus-sparse ordering the neural surrogate shows on voxel fields. By observable type: the integrating quantities are emulated well (time-average 0.866, AUC 0.866) and the pointwise ones are not (final 0.046, max 0.610), with four of six final-value observables below zero. A single time point of a stochastic model, or an extremum over one, is dominated by realisation noise; averaging over 101 time points cancels it. Nineteen of 24 pass R² ≥ 0.5.

**Which indices are quantitative.** `helpers/compare_sobol.py` recomputes the ranking on the observables that pass the emulator threshold, as a check on how much of each index rests on poorly emulated observables. Under subsampling of the 100 runs, only the leading index (`sigmoidb`, S_T = 0.456) keeps its rank; `keil8`, `init_ec` and `km2il10` (0.13-0.17) stay within the subsampling spread without a stable order. Treat only the leading index as quantitative and the rest as a screen.

**Observable.** `datafiles/mean_concentration.txt` holds, per cytokine and output step, the mean of the field sampled at the centres of mass of all cells - not a volume average. These cell-sampled trajectories are reduced to `[final, mean, max, AUC]` per cytokine - 24 scalars - defined once in `smore/observables.py` (`summarize_observable`, `FEATURE_NAMES`); do not redefine them anywhere else. The surrogate-in-the-loop experiment instead averages the full predicted and ABM fields over the volume, in both arms.

**Stage one of SMoRe ParS.** The trajectory representation that is mapped and inverted is the set of summary observables above. A saturating-logistic surface (plateau, rate, inflection) is still fitted in `smore_pars.py`, in units of each trajectory's maximum (at physical scale, ~1e-9, the optimiser does not move from its starting point), but only as a diagnostic: on the ABM trajectories its median R² is 0.52 for IL-8, 0.15 for TGF-β and about 0 for the other four cytokines, so its parameters do not describe these trajectories. On the *E. coli* growth curves, where it fits (median R² 0.986), it is used as the stage-one representation.

**Calibration scope.** SMoRe ParS recovers the top-k from the Sobol ranking. Parameters that do not move the observable are not identifiable and add an unconstrained direction to the fit. In this sweep, including the non-identifiable `init_ec` did not measurably degrade recovery of the other calibrated parameters (see the results table), so the cost of choosing targets by sensitivity alone was one unconstrained parameter. This follows Jain 2022 (few parameters) -> Bergman 2024 (higher-dimensional).

**Endothelium is frozen.** `init_ec` affects IL-8 only through the number of constitutive sources, so in the cell-sampled observables it is collinear with `keil8`: multiplying one and dividing the other by the same factor leaves them essentially unchanged. It is therefore sensitive (S_T = 0.139, third of ten) but recovered no better than its mean (R² = +0.045). `helpers/ridge_raw.py` confirms the ridge on a GP emulator of the final cell-sampled IL-8 (cross-validated R² 0.987): the observable correlates with the product `init_ec` × `keil8` at |r| = 0.984, against 0.568 and 0.817 for the two factors separately, and its variation along curves of constant product is 0.18 of its total variation. From the volume-averaged IL-8 used in the loop, `init_ec` is recovered at +0.77, so the non-identifiability depends on how the concentration is measured. It arises from the ABM configuration inherited from Korkmaz et al., not from the calibration method.

---

## Results this pipeline produced

| Stage | Outcome |
|---|---|
| Sobol, all 24 observables | `sigmoidb` 0.456 clearly first; `keil8` 0.172, `init_ec` 0.139, `km2il10` 0.126 without a stable order |
| Recovery (leave-one-out, all ten jointly) | `keil8` +0.803, `sigmoidb` +0.786, `km2tgf` +0.575, `km1il6` +0.548, `km2il10` +0.519 |
| Recovery, non-identifiable | `init_ec` +0.045 despite rank 3; `lnril8`, `init_n`, `init_f`, `init_m` negative |
| Recovery target choice | Top 5 by Sobol vs `init_ec` replaced by `km1il6`: mean nRMSE 0.195 vs 0.198, no measurable difference |
| Ridge | Final cell-sampled IL-8 tracks `init_ec` × `keil8` at \|r\| = 0.984 (flatness 0.18) |
| Surrogate in the loop | Field R² 0.93-0.98 across seeds, `keil8` recovery +0.19 to +0.62, against +0.93 from the ABM's own volume-averaged IL-8 observables |
| External validation (*E. coli*) | Logistic surface fits the 870 curves closely (median 0.986, against 0.52 on ABM IL-8); no input recovered (leave-one-out on a random 500-curve subsample) |

The surrogate-in-the-loop row is the one worth internalising before reusing this code: the surrogate reproduces the trajectories almost perfectly, yet the parameter is recovered far worse than from the ABM, and field accuracy and recovery do not rank the seeds in the same order.
A surrogate intended for calibration has to be validated on a recovery task, not only on a field-accuracy metric, and across seeds rather than at one.

---

## Status

Validated end-to-end on a synthetic sweep with known θ-dependence: Sobol recovered exactly the injected drivers, and SMoRe ParS recovered them with positive R². The 100-run real sweep has been executed and the results above come from it.
