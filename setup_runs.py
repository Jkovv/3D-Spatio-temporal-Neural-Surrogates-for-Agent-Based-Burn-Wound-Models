#!/usr/bin/env python3
# combi3D/Simulation/setup_runs.py
#
# Clean, deterministic LHS manifest generator for the SMoRe ParS sweep.
#
# WHY THIS EXISTS (and why it replaces the old generator):
#   The previous sweep desynced: manifest.json and the per-run override files
#   were produced by two different sampling passes, some overrides fell outside
#   the stated bounds, only runs 1-4 got override files (5-10 ran on baseline),
#   and the initial cell counts never varied. Root cause: the manifest and the
#   things that actually drove the sims were SEPARATE artefacts that had to be
#   kept in sync by hand.
#
#   This generator makes the manifest the SINGLE SOURCE OF TRUTH. It samples
#   every parameter ONCE, in one pass, with one seeded LHS design, and writes
#   exactly one manifest.json. run_sweep.py then explodes that manifest into the
#   per-run JSONs the simulation reads. There is no second artefact to drift.
#
# METHOD (matches the SMoRe ParS literature):
#   Latin Hypercube Sampling via scipy.stats.qmc.LatinHypercube, which is
#   deterministic under a fixed seed and independent of SALib (so the later
#   Sobol/SALib step does not share RNG state with sweep generation). Each
#   parameter is sampled in [low, high] from the bounds table below; bounds are
#   +/-30% around the baseline unless noted (cell counts are integer-valued).
#
# INTENDED PIPELINE (the scientifically correct order):
#   1. setup_runs.py  -> manifest.json           (sweep ALL parameters)
#   2. run_sweep.py   -> per-run sims            (one theta per trajectory)
#   3. Sobol (emulator-based) on the runs        -> rank parameter sensitivity
#   4. SMoRe ParS on the TOP-K sensitive params  -> recover theta_ABM
#   Calibrating only the top-k (not all 10) is deliberate: parameters that do
#   not move the outputs are not identifiable and would depress recovery scores
#   for reasons that have nothing to do with the method. Sobol picks which
#   parameters are worth calibrating.

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import qmc


# ── Parameter definitions: baseline + bounds. ───────────────────────────────
# "kind" mirrors param_loader: "int_count" parameters are rounded to ints.
# Bounds here are the ones from the supervisor's manifest (±30% on rates,
# ±30% on cell counts, sigmoidb ±30%, lnril8 ±30%). Edit in ONE place.
PARAM_SPEC = {
    "keil8":    {"baseline": 3.9000000000000005e-08, "low": 2.7300000000000003e-08, "high": 5.070000000000001e-08,  "kind": "float", "description": "Endothelial cell IL-8 secretion rate"},
    "km1il6":   {"baseline": 4.1666666666666676e-08, "low": 2.916666666666667e-08,  "high": 5.416666666666668e-08,  "kind": "float", "description": "M1 macrophage IL-6 secretion rate"},
    "km2il10":  {"baseline": 7.500000000000003e-09,  "low": 5.2500000000000015e-09, "high": 9.750000000000005e-09,  "kind": "float", "description": "M2 macrophage IL-10 secretion rate"},
    "km2tgf":   {"baseline": 4.666666666666669e-08,  "low": 3.266666666666668e-08,  "high": 6.06666666666667e-08,   "kind": "float", "description": "M2 macrophage TGF-beta1 secretion rate"},
    "lnril8":   {"baseline": 0.25,                   "low": 0.175,                  "high": 0.325,                  "kind": "float", "description": "Neutrophil recruitment: IL-8 sigmoid weight"},
    "sigmoidb": {"baseline": 4.0,                    "low": 2.8,                    "high": 5.2,                    "kind": "float", "description": "Sigmoid steepness parameter b"},
    "init_ec":  {"baseline": 100.0,                  "low": 70.0,                   "high": 130.0,                  "kind": "int_count", "description": "Initial endothelial cell count"},
    "init_n":   {"baseline": 110.0,                  "low": 77.0,                   "high": 143.0,                  "kind": "int_count", "description": "Initial neutrophil count"},
    "init_m":   {"baseline": 100.0,                  "low": 70.0,                   "high": 130.0,                  "kind": "int_count", "description": "Initial monocyte count"},
    "init_f":   {"baseline": 10.0,                   "low": 7.0,                    "high": 13.0,                   "kind": "int_count", "description": "Initial fibroblast count"},
}

# Canonical ordering (LHS columns map to this order, so a fixed seed gives a
# fixed design).
PARAM_ORDER = list(PARAM_SPEC.keys())


def generate(n_runs, seed, scramble=True):
    d = len(PARAM_ORDER)
    sampler = qmc.LatinHypercube(d=d, seed=seed, scramble=scramble)
    unit = sampler.random(n=n_runs)          # (n_runs, d) in [0,1)

    lows  = np.array([PARAM_SPEC[p]["low"]  for p in PARAM_ORDER])
    highs = np.array([PARAM_SPEC[p]["high"] for p in PARAM_ORDER])
    scaled = qmc.scale(unit, lows, highs)    # (n_runs, d) in [low, high]

    runs = []
    for i in range(n_runs):
        params = {}
        for j, name in enumerate(PARAM_ORDER):
            val = float(scaled[i, j])
            if PARAM_SPEC[name]["kind"] == "int_count":
                val = int(round(val))
            params[name] = val
        runs.append({"run_id": f"run_{i+1:04d}", "params": params})
    return runs


def build_manifest(n_runs, seed, scramble):
    runs = generate(n_runs, seed, scramble)
    return {
        "n_runs": n_runs,
        "sampling_method": "lhs_scipy_qmc",
        "scramble": scramble,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "param_names": list(PARAM_ORDER),
        "baselines": {p: PARAM_SPEC[p]["baseline"] for p in PARAM_ORDER},
        "bounds": {p: {"low": PARAM_SPEC[p]["low"], "high": PARAM_SPEC[p]["high"],
                       "description": PARAM_SPEC[p]["description"]}
                   for p in PARAM_ORDER},
        "runs": runs,
    }


def validate(manifest):
    """Self-check: every sampled value must lie within its stated bounds, and
    int_count params must be integers. This is the guard the old pipeline
    lacked (it let sigmoidb=12.5 through against bounds 2.8-5.2)."""
    problems = []
    for run in manifest["runs"]:
        for name, val in run["params"].items():
            lo = PARAM_SPEC[name]["low"]; hi = PARAM_SPEC[name]["high"]
            if not (lo - 1e-12 <= val <= hi + 1e-12):
                problems.append(f"{run['run_id']}.{name}={val} outside [{lo},{hi}]")
            if PARAM_SPEC[name]["kind"] == "int_count" and int(val) != val:
                problems.append(f"{run['run_id']}.{name}={val} not integer")
    if problems:
        raise ValueError("Manifest validation FAILED:\n  " + "\n  ".join(problems))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-runs", type=int, default=10,
                    help="Number of LHS runs (PoC default 10).")
    ap.add_argument("--seed", type=int, default=42,
                    help="LHS seed (fixed -> reproducible manifest).")
    ap.add_argument("--no-scramble", action="store_true",
                    help="Disable LHS scrambling (default: scrambled).")
    ap.add_argument("--out", default="manifest.json",
                    help="Output manifest path.")
    args = ap.parse_args()

    manifest = build_manifest(args.n_runs, args.seed, not args.no_scramble)
    validate(manifest)

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[setup_runs] wrote {args.out}")
    print(f"[setup_runs] {args.n_runs} runs x {len(PARAM_ORDER)} params, "
          f"seed={args.seed}, method=lhs_scipy_qmc")
    print(f"[setup_runs] validation: all values within bounds OK")
    print(f"[setup_runs] next: python run_sweep.py --manifest {args.out} "
          f"--mode local --cc3d-run <path>")


if __name__ == "__main__":
    main()
