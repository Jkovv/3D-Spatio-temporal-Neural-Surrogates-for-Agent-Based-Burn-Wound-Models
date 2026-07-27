# SMoRe ParS calibration pipeline for the 3D burn ABM
## Layout

```
smores/
├── manifest.json              # single source of truth: 100 runs × 10 params, LHS seed 42
├── smore/                     # calibration package
│   ├── observables.py         # (θ_ABM, mean-concentration trajectory) per run
│   ├── sensitivity.py         # GP emulator -> Sobol ranking
│   ├── smore_pars.py          # surface fit, θ_ABM↔θ_SM map, leave-one-out recovery
│   ├── run_calibration.py     # orchestrates sweep -> Sobol -> SMoRe ParS
│   ├── run_calibration_surrogate.py   # same, with surrogate-predicted observables
│   ├── compare_observables.py
│   └── spatial_observables.py
├── helpers/ # helper files for experiments mentioned in the manusctipt
│   ...
└── sweep/
    └── outputs/run_0001 … run_0100/
        ├── params.json                        # θ for this run (ground truth)
        ├── datafiles/mean_concentration.txt   # 101 rows: MCS + 6 means + 6 SDs
        ├── datafiles/cellcount.txt
        └── LatticeData/{Cyto,Cell}Step_*.npz  # 50³ fields, 6 cytokines, 101 steps
      ...
```

`manifest.json` carries `param_names`, `bounds` (per parameter `low`/`high`
plus a description), `baselines`, and the per-run vectors under `runs`.
`params.json` in each run directory nests the vector under a `params` key and
is the pairing key between θ and trajectory.

---

## Parameter injection

Per-run-directory staging, matching the original sweep's structure and safer
than an environment variable, since it does not depend on CC3D passing the
environment through to the steppable process.

`run_sweep.py` stages `sweep/runs/<run_id>/Simulation/` with a full copy of the
code plus a validated `params.json`; `param_loader.py` reads that local file
(or `$SMORE_PARAMS` if set); CC3D writes to `sweep/outputs/<run_id>/`, outside
the run directory as CC3D requires, and `params.json` is copied there.

CC3D is launched with:

```
<cc3d_python> -m cc3d.run_script --input=<run>/combi3D.cc3d --output-dir=<out>
```

---

## Running it

```bash
# local sanity check, no CC3D required
python verify.py                      # must print ALL CHECKS PASSED

# one-time install (~1–2 h)
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
```

`emulator_cv.py` and `compare_sobol.py` import `smore/observables.py` and
`smore/sensitivity.py` rather than reimplementing anything, so the emulator
they score is the emulator the indices stand on. Run `emulator_cv.py` before
`compare_sobol.py` - the second consumes the first's CSV.

Note that `compare_sobol.py` takes `--n-saltelli` (default 1024). Pass the same
base sample the production `sensitivity.py` run used, otherwise its "All"
column will not reproduce the main Sobol table.

---

## Method notes

**Sampling.** Latin hypercube via `scipy.stats.qmc.LatinHypercube`,
deterministic under a fixed seed and independent of SALib, so the Sobol step
shares no RNG state with sweep generation.

**Sensitivity.** A Gaussian process is fitted per observable on the 100 real
runs and Sobol indices are computed on a dense Saltelli sample of that
emulator; a direct Saltelli design on the ABM would need thousands of 50³
trajectories. `sensitivity._fit_gp` standardises θ and y internally and returns
`(predict, gp)`.

**Emulator quality is not uniform, and this bounds the ranking.** Cross-
validating the 24 emulators (`helpers/emulator_cv.py`, pooled out-of-fold R²,
5-fold, refitted per fold) gives a mean of 0.590 with a range from −0.421 to
+0.991. The variation is structured along two axes at once. By cytokine: IL-8
reaches 0.984 while IL-1β reaches 0.387, the same dense-versus-sparse ordering
the neural surrogate shows on voxel fields. By observable type: the
integrating quantities are emulated well (mean 0.821, AUC 0.826) and the
pointwise ones are not (final 0.111, max 0.601), with five of six final-value
observables at or below zero. A single time point of a stochastic model, or an
extremum over one, is dominated by realisation noise; averaging over 101 time
points cancels it. Seventeen of 24 pass R² ≥ 0.5.

**Which indices are quantitative.** Recomputing the ranking on those 17
(`helpers/compare_sobol.py`) leaves the three leaders unmoved - `sigmoidb`,
`keil8`, `init_ec` - while `init_m`, `init_n`, `init_f` and `lnril8` collapse to
within 0.002 of zero. Their apparent influence was carried by observables the
emulator was not reproducing, which is also why `init_m`'s index failed to
settle under subsampling. Treat only the three leaders as quantitative.

**Calibration scope.** SMoRe ParS recovers only the top-k from the Sobol
ranking. Parameters that do not move the observable are not identifiable, and
calibrating them injects an unconstrained direction that depresses recovery for
everything else. This follows Jain 2022 (few parameters) -> Bergman 2024
(higher-dimensional).

**Observable.** Per-cytokine volume-averaged concentration time series from
`datafiles/mean_concentration.txt`, reduced to `[final, mean, max, AUC]` per
cytokine - 24 scalars. Defined once in `smore/observables.py`
(`summarize_observable`, `FEATURE_NAMES`); do not redefine them anywhere else.

**Endothelium is frozen.** `init_ec` affects IL-8 only through the number of
constitutive sources, so it is collinear with `keil8`: multiplying one and
dividing the other by the same factor leaves the trajectory essentially
unchanged. It is therefore sensitive (S_T = 0.109, third of ten, and third
again on the filtered ranking) but not recoverable (R² = −0.454). This is a
property of the ABM configuration inherited from Korkmaz et al., not of the
calibration method. It is the single case in the sweep where influence and
invertibility come apart.

---

## Results this pipeline produced

| Stage | Outcome |
|---|---|
| Sobol, all 24 observables | `sigmoidb` 0.457, `keil8` 0.216 lead; `init_ec` third at 0.109 |
| Sobol, 17 well-emulated | Same three leaders; `init_*` and `lnril8` collapse to ≈0 |
| Recovery (leave-one-out) | `keil8` +0.833, `km2il10` +0.481, `sigmoidb` +0.439, `km1il6` +0.374 |
| Recovery, non-identifiable | All four `init_*` negative; `init_ec` −0.454 despite rank 3 |
| Surrogate in the loop | Field R² 0.987–0.998 across seeds, `keil8` recovery +0.191 to −0.077 |
| External validation (*E. coli*) | Surface fits better than on the ABM (median 0.986); no input recovered |

The surrogate-in-the-loop row is the one worth internalising before reusing this code: field accuracy and parameter recoverability do not move together, and the seed with the most accurate fields recovered the parameter least well.
A surrogate intended for calibration has to be validated on a recovery task, not only on a field-accuracy metric, and across seeds rather than at one.

---

## Status

Validated end-to-end on a synthetic sweep with known θ-dependence: Sobol
recovered exactly the injected drivers, and SMoRe ParS recovered them with
positive R². The 100-run real sweep has been executed and the results above
come from it.
