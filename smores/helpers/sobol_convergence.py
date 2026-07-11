#!/usr/bin/env python3
"""
helpers/sobol_convergence.py

Shows that the Sobol sensitivity indices have converged at N=100 runs, so the
sweep size is sufficient. Recomputes the emulator-based Sobol analysis on
growing sub-samples of the sweep (e.g. 60, 70, 80, 90, 100 runs) and reports
how the top-parameter total-order indices and their ranking stabilise.

WHY THIS EXISTS
---------------
The paper states the sweep is a proof-of-concept in size and that bootstrap
confidence intervals are used to check convergence. This turns that into a
reproducible check: if the ranking and the leading ST values stop changing as
runs are added, 100 is enough. If they were still moving, we would need more.

It uses the SAME emulator_sobol routine as the main calibration, so the numbers
are directly comparable to calibration_results_*.json.

USAGE
-----
    python helpers/sobol_convergence.py \
        --sim-root sweep/outputs --manifest manifest.json \
        --subsamples 60 70 80 90 100 --n-saltelli 512

Writes helpers/sobol_convergence_report.txt.

Note: run from the smores/ directory (so `smore/` is importable), or the script
adds it to sys.path itself.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# make smore/ importable whether run from smores/ or helpers/
HERE = Path(__file__).resolve().parent
SMORE_DIR = HERE.parent / "smore"
sys.path.insert(0, str(SMORE_DIR))

from observables import load_sweep, summarize_observable, FEATURE_NAMES  # noqa
from sensitivity import emulator_sobol  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--subsamples", type=int, nargs="+",
                    default=[60, 70, 80, 90, 100],
                    help="Run counts to evaluate convergence at.")
    ap.add_argument("--n-saltelli", type=int, default=512,
                    help="Saltelli base sample (smaller than main run is fine "
                         "for a convergence trend; default 512).")
    ap.add_argument("--top", type=int, default=5,
                    help="How many leading parameters to track.")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    names = man["param_names"]
    bounds = man["bounds"]

    theta_all, Y_all, ids, t = load_sweep(args.sim_root, names)
    n_total = theta_all.shape[0]
    print(f"[convergence] loaded {n_total} runs, {len(names)} params")

    subs = [n for n in args.subsamples if n <= n_total]
    if subs[-1] != n_total:
        subs.append(n_total)

    # For each sub-sample size, compute the mean-ST ranking. Use the FIRST n
    # runs (LHS order is already space-filling, so a prefix is a valid design).
    results = {}
    for n in subs:
        feats = summarize_observable(Y_all[:n])
        sob = emulator_sobol(theta_all[:n], feats, FEATURE_NAMES, names, bounds,
                             n_saltelli=args.n_saltelli)
        st_by_param = {r["param"]: r["ST_mean"] for r in sob["ranking"]}
        order = [r["param"] for r in sob["ranking"]]
        results[n] = {"st": st_by_param, "order": order}
        print(f"[convergence] N={n:3d}: top-{args.top} = {order[:args.top]}")

    # Reference ranking at full N.
    ref_order = results[n_total]["order"]
    top_params = ref_order[:args.top]

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 72)
    emit("SOBOL CONVERGENCE  (are 100 runs enough?)")
    emit("=" * 72)
    emit(f"sub-samples: {subs}   n_saltelli={args.n_saltelli}")
    emit(f"tracking top-{args.top} parameters at full N={n_total}: {top_params}")
    emit("")

    # Table: ST of each top parameter as N grows.
    header = "param".ljust(11) + "".join(f"N={n:<8}" for n in subs)
    emit(header)
    emit("-" * len(header))
    for p in top_params:
        row = p.ljust(11)
        for n in subs:
            row += f"{results[n]['st'].get(p, float('nan')):<10.3f}"
        emit(row)

    emit("")
    # Ranking stability: how many of the top-k are the same set as full-N.
    emit("Ranking stability (top-{} set overlap with full N):".format(args.top))
    ref_set = set(top_params)
    for n in subs:
        s = set(results[n]["order"][:args.top])
        overlap = len(s & ref_set)
        same_order = results[n]["order"][:args.top] == top_params
        tag = "identical order" if same_order else f"{overlap}/{args.top} same set"
        emit(f"  N={n:3d}: {tag}")

    emit("")
    # Max ST drift over the last two sub-samples (a convergence proxy).
    if len(subs) >= 2:
        n1, n2 = subs[-2], subs[-1]
        drift = max(abs(results[n2]["st"][p] - results[n1]["st"][p])
                    for p in top_params)
        emit(f"Max |ST(N={n2}) - ST(N={n1})| over top-{args.top}: {drift:.3f}")
        if drift < 0.05:
            emit("  -> leading indices change by <0.05 over the last step:"
                 " converged.")
        else:
            emit("  -> still moving; consider more runs.")
    emit("=" * 72)

    out = HERE / "sobol_convergence_report.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()