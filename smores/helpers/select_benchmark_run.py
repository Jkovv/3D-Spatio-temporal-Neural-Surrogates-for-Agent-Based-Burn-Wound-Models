#!/usr/bin/env python3
"""
helpers/select_benchmark_run.py

Proves which sweep run is the most representative (central) point in the
LHS parameter space, so the surrogate benchmark run is chosen by a defensible
criterion rather than arbitrarily.

WHY THIS EXISTS
---------------
The surrogate accuracy tables in the paper are computed on ONE sweep run (the
thesis protocol: one run, seeds 1/42/100). Which run that is matters: a run
sitting in a corner of the parameter space (e.g. near-minimal IL-10 kinetics)
would make the surrogate look artificially worse on the sparse cytokine, not
because of the surrogate but because that run barely has any IL-10. Choosing
the run nearest the CENTRE of the sampled space makes the benchmark reflect
typical dynamics.

WHAT "CENTRAL" MEANS HERE
-------------------------
Each parameter is min-max normalised to [0,1] across the sweep. A run's
"centrality score" is the mean absolute distance of its normalised parameters
from 0.5 (the centre). 0.0 = dead centre on every parameter; ~0.5 = extreme
corner. The most central run has the smallest score. We also report, per run,
how many parameters are "extreme" (normalised <0.15 or >0.85), since a low
mean can still hide one extreme axis.

USAGE
-----
    python helpers/select_benchmark_run.py                 # uses seed 42, N=100
    python helpers/select_benchmark_run.py --n-runs 100 --seed 42
    python helpers/select_benchmark_run.py --manifest ../manifest.json

Writes helpers/benchmark_run_justification.txt with the ranking and the chosen
run, ready to cite in the Methods.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_manifest(manifest_path, n_runs, seed):
    """Load an existing manifest, or regenerate the identical one (seed, N)."""
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            return json.load(f), f"loaded {manifest_path}"
    # regenerate the SAME manifest deterministically via setup_runs.py
    here = Path(__file__).resolve().parent
    setup = here.parent / "setup_runs.py"
    tmp = here / "_tmp_manifest.json"
    subprocess.run(
        [sys.executable, str(setup), "--n-runs", str(n_runs),
         "--seed", str(seed), "--out", str(tmp)],
        check=True, capture_output=True,
    )
    with open(tmp) as f:
        man = json.load(f)
    os.remove(tmp)
    return man, f"regenerated (seed={seed}, N={n_runs})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="Path to an existing manifest.json (else regenerate).")
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extreme-lo", type=float, default=0.15)
    ap.add_argument("--extreme-hi", type=float, default=0.85)
    ap.add_argument("--top", type=int, default=10,
                    help="How many central runs to list.")
    args = ap.parse_args()

    manifest, source = load_manifest(args.manifest, args.n_runs, args.seed)
    runs = manifest["runs"]
    names = [r["run_id"] for r in runs]
    pnames = list(runs[0]["params"].keys())

    P = np.array([[r["params"][p] for p in pnames] for r in runs], dtype=float)
    Pmin, Pmax = P.min(0), P.max(0)
    Pn = (P - Pmin) / (Pmax - Pmin + 1e-12)          # normalise each param

    centrality = np.abs(Pn - 0.5).mean(1)             # 0 = centre
    n_extreme = ((Pn < args.extreme_lo) | (Pn > args.extreme_hi)).sum(1)
    order = np.argsort(centrality)

    # Prefer the most central run that ALSO has zero extreme axes, if one
    # exists among the top candidates; otherwise the most central overall.
    chosen = None
    for j in order:
        if n_extreme[j] == 0:
            chosen = j
            break
    if chosen is None:
        chosen = order[0]

    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 66)
    emit("BENCHMARK RUN SELECTION — centrality in LHS parameter space")
    emit("=" * 66)
    emit(f"manifest source : {source}")
    emit(f"runs            : {len(names)}")
    emit(f"parameters      : {len(pnames)}  ({', '.join(pnames)})")
    emit(f"extreme cutoff  : normalised <{args.extreme_lo} or >{args.extreme_hi}")
    emit("")
    emit(f"centrality = mean |normalised_param - 0.5|  (0=centre, ~0.5=corner)")
    emit("")
    emit(f"{'rank':>4}  {'run':>10}  {'centrality':>11}  {'#extreme_params':>15}")
    emit("-" * 48)
    for rank, j in enumerate(order[:args.top], start=1):
        mark = "  <-- CHOSEN" if j == chosen else ""
        emit(f"{rank:>4}  {names[j]:>10}  {centrality[j]:>11.3f}  "
             f"{n_extreme[j]:>15}{mark}")
    emit("")
    emit(f"CHOSEN BENCHMARK RUN: {names[chosen]}")
    emit(f"  centrality score : {centrality[chosen]:.3f} "
         f"(rank {list(order).index(chosen)+1}/{len(names)})")
    emit(f"  extreme params   : {n_extreme[chosen]}")
    emit("")
    emit(f"  normalised parameter vector (0=sweep min, 1=sweep max):")
    for p, v in zip(pnames, Pn[chosen]):
        flag = "  <-- extreme" if (v < args.extreme_lo or v > args.extreme_hi) else ""
        emit(f"    {p:10s}: {v:.3f}{flag}")
    emit("")
    emit("PAPER-READY JUSTIFICATION:")
    emit(f"  The surrogate benchmark uses {names[chosen]}, the sweep point")
    emit(f"  nearest the centre of the sampled parameter space (centrality")
    emit(f"  {centrality[chosen]:.3f}, {n_extreme[chosen]} extreme parameters), so that")
    emit(f"  reported accuracy reflects typical cytokine dynamics rather than")
    emit(f"  an extreme corner of the parameter space.")
    emit("=" * 66)

    out = Path(__file__).resolve().parent / "benchmark_run_justification.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()