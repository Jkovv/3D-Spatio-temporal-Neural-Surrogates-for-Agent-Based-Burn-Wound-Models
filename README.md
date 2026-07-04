# combi3D - SMoRe ParS calibration pipeline

End-to-end, reproducible pipeline replacing the old desynced sweep. Single
source of truth (one manifest), fail-loud parameter injection, sensitivity-first
calibration on the top-k identifiable parameters.

## What was broken before (and is now fixed)

The previous sweep desynced. After inspecting the original `sweep/setup_runs.py`,
the failure modes were:

1. **Generated `.py` overrides, written with `repr()` of floats.** Each run got
   a `transcriptomics_overrides.py` text-generated from a manifest. If the
   manifest used to generate overrides for runs 1–4 differed from the one used
   later (or the generator was interrupted mid-loop), the overrides and the
   manifest desynced - which is exactly what happened (overrides existed only
   for runs 1–4; 5–10 ran on baseline).
2. **Integer cell counts depended on a SEPARATE `param_bounds.json`.** The
   `integer:true` flags were read from a different file than the manifest; if it
   was absent, `integer_params` was empty and cell counts were written as
   floats (`init_ec = 101.43…`), which breaks `range()`.
3. **Step count injected by regex into a copied `combi3D.py`.** If the pattern
   didn't match it only warned and the run used the default step count.
4. **No name validation.** Any key in the manifest was written blindly.

All four were silent: a misconfigured run still finished and produced output.

Now: the manifest is the **only** source of truth; cell-count integer handling
and bounds live with the parameters in one place; the loader **crashes** on
unknown names, missing files, or non-finite values instead of falling back.

## Parameter injection: per-run-directory layout

Matching the original sweep's structure (and safer than an env var, since it
does not depend on CC3D passing the environment through to the steppable
process):

- `run_sweep.py` stages `sweep/runs/<run_id>/Simulation/` with a full copy of
  the code plus a validated `params.json`.
- `param_loader.py` reads that local `params.json` (or `$SMORE_PARAMS` if set).
- CC3D output goes to `sweep/outputs/<run_id>/` (outside the run dir, as CC3D
  requires), and `params.json` is copied there so SMoRe ParS can pair θ with the
  trajectory.

CC3D is launched with the real command:
`<cc3d_python> -m cc3d.run_script --input=<run>/combi3D.cc3d --output-dir=<out>`

## Files

Simulation core (drop into `combi3D/Simulation/`):
- `setup_runs.py` - LHS manifest generator (scipy.qmc, deterministic).
- `manifest.json` - generated sweep (10 runs × 10 params, seed 42).
- `run_sweep.py` - stages per-run dirs; runs CC3D (local/slurm).
- `param_loader.py` - reads local `params.json` / `$SMORE_PARAMS`, validates, caches.
- `transcriptomics_overrides.py` - applies the per-run vector (scope-aware).
- `params_*.py`, `combi3D.py`, `combi3DSteppables.py`, `solver3D.py`, `variablevals3D.py`, `combi3D.cc3d` - the simulation. (`variablevals3D.py` is loaded by `combi3D.cc3d` as a Resource and must be present.)

Calibration package (`combi3D/Simulation/smore/`):
- `observables.py` - loads (θ_ABM, mean-concentration trajectory) per run.
- `sensitivity.py` - emulator-based Sobol → parameter ranking.
- `smore_pars.py` - surrogate fit, GP θ_ABM↔θ_SM mapping, leave-one-out recovery.
- `run_calibration.py` - orchestrates sweep → Sobol → SMoRe ParS.

Self-check:
- `verify.py` - runs the whole pipeline without CC3D and prints [ok]/[FAIL]
  for every stage. 

## Run order (Snellius)

```bash
# 0. upload + unzip the pipeline there, then:

# 1. install miniconda + CC3D + FiPy + calibration deps (ONE TIME, ~1-2h)
sbatch install_cc3d.slurm
tail -f install_<jobid>.out          # wait for "DONE"

# 2. single test run - confirms CC3D produces mean_concentration.txt
sbatch test_run.slurm
tail -f testrun_<jobid>.out           # wait for "SUCCESS"

# 3. full sweep (10 runs as a SLURM array). test_run.slurm already staged the
#    run dirs and wrote sweep/sweep_array.sh, so just submit it:
sbatch ../sweep/sweep_array.sh

# 4. calibrate once all runs finish: sensitivity FIRST, then SMoRe ParS top-k
python smore/run_calibration.py --sim-root ../sweep/outputs \
    --manifest manifest.json --top-k 5 --out calibration_results.json
```

Paths assume `SCRATCH=/gpfs/scratch1/shared/jkowalczuk` and env name `cc3d`.
Edit the variables at the top of `install_cc3d.slurm` / `test_run.slurm` if
yours differ. Python is pinned to 3.10 to match the supervisor's working
environment (his `.pyc` files are cpython-310).

### Local sanity check (no CC3D, any machine)

```bash
python verify.py     # must print ALL CHECKS PASSED
```

## Method notes 

- **Sampling**: Latin Hypercube via `scipy.stats.qmc.LatinHypercube`,
  deterministic under a fixed seed, independent of SALib so the Sobol step
  shares no RNG state with sweep generation.
- **Sensitivity**: at N=10 a full Saltelli design is infeasible; a GP emulator
  is fit on the real runs and Sobol is computed on a dense emulator sample.
  Indices are **indicative** (wide CIs), used to rank/select, not as final
  variance attributions. Larger sweep (phase 2) gives stable indices.
- **Calibration scope**: SMoRe ParS recovers only the **top-k** parameters from
  the Sobol ranking. Parameters that do not move the observable are not
  identifiable; calibrating them would depress recovery for reasons unrelated
  to the method. This matches Jain 2022 (few params) → Bergman 2024 (high-dim).
- **Observable**: per-cytokine mean concentration time series (from
  `datafiles/mean_concentration.txt`), summarised as [final, mean, max, AUC]
  per cytokine.
- **Caveat - endothelial is frozen**: `init_ec` affects IL-8 only through the
  number of IL-8 sources, so its effect is partly collinear with `keil8`.
  Note this when reading the Sobol ranking.

## Status

Validated end-to-end on a synthetic sweep with known θ-dependence: Sobol
recovered exactly the injected drivers and SMoRe ParS recovered them with
positive R². On real CC3D output the numbers will differ; the machinery is
correct. First real step: run one simulation locally and confirm CC3D picks up
`$SMORE_PARAMS` before launching the full sweep.
