#!/usr/bin/env python3
"""
helpers/analyze_identifiability.py

Formally documents the central methodological finding of the calibration study:
SENSITIVITY and IDENTIFIABILITY are different properties. A parameter can have a
high Sobol total-order index (it influences the observables) yet be
non-identifiable (it cannot be recovered from them). init_ec is exactly this
case in the 3D burn model.

WHY THIS EXISTS
---------------
The paper claims the Sobol ranking acts as a filter for calibration. This script
turns that claim into a reproducible, paper-ready table by cross-tabulating, for
every parameter:
    - its Sobol total-order index (ST_mean)  -> "does it matter?"
    - its leave-one-out recovery R2          -> "can we recover it?"
and flagging the mismatch cases (high ST, R2 < 0). It then quantifies how much
recovery improves when the non-identifiable init_ec is swapped out of the
top-5 for the identifiable km2tgf.

INPUTS (produced by run_calibration.py)
---------------------------------------
    calibration_results_topk10.json           (full ranking + recovery)
    calibration_results_topk5_sobol.json       (top-5 by Sobol, incl. init_ec)
    calibration_results_topk5_identifiable.json (5 identifiable, incl. km2tgf)

USAGE
-----
    python helpers/analyze_identifiability.py \
        --topk10 calibration_results_topk10.json \
        --topk5-sobol calibration_results_topk5_sobol.json \
        --topk5-ident calibration_results_topk5_identifiable.json

Writes helpers/identifiability_report.txt (paper-ready).
"""

import argparse
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk10", required=True,
                    help="Full-ranking calibration bundle (all 10 params).")
    ap.add_argument("--topk5-sobol", default=None,
                    help="Top-5-by-Sobol bundle (includes init_ec).")
    ap.add_argument("--topk5-ident", default=None,
                    help="5-identifiable bundle (includes km2tgf).")
    ap.add_argument("--r2-floor", type=float, default=0.0,
                    help="R2 threshold below which a parameter is called "
                         "non-identifiable (default 0.0).")
    args = ap.parse_args()

    full = load(args.topk10)
    ranking = full["sobol"]["ranking"]
    r2 = full["recovery"]["r2_per_param"]
    nrmse = full["recovery"]["nrmse_per_param"]

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 70)
    emit("IDENTIFIABILITY ANALYSIS - sensitivity is not identifiability")
    emit("=" * 70)
    emit(f"source: {args.topk10}")
    emit(f"non-identifiable threshold: recovery R2 < {args.r2_floor}")
    emit("")
    emit(f"{'rank':>4}  {'param':>10}  {'Sobol_ST':>9}  {'recovery_R2':>11}  "
         f"{'nRMSE':>7}  {'verdict':>28}")
    emit("-" * 78)

    sensitive_not_identifiable = []
    for i, r in enumerate(ranking, 1):
        p = r["param"]
        st = r["ST_mean"]
        rr = r2[p]
        nn = nrmse[p]
        # verdict logic
        sensitive = st > 0.10          # meaningfully above zero
        identifiable = rr >= args.r2_floor
        if sensitive and not identifiable:
            verdict = "SENSITIVE but NOT identifiable"
            sensitive_not_identifiable.append(p)
        elif sensitive and identifiable:
            verdict = "sensitive & identifiable"
        elif not sensitive and identifiable:
            verdict = "weak but identifiable"
        else:
            verdict = "neither (correctly filtered)"
        emit(f"{i:>4}  {p:>10}  {st:>9.3f}  {rr:>+11.3f}  {nn:>7.3f}  {verdict:>28}")

    emit("")
    emit("KEY FINDING:")
    if sensitive_not_identifiable:
        for p in sensitive_not_identifiable:
            emit(f"  * {p}: high Sobol ST (influences observables) but recovery "
                 f"R2={r2[p]:+.3f} (< {args.r2_floor}).")
        emit("    -> A high total-order index does NOT guarantee identifiability.")
        emit("    -> Sensitivity ranks influence; recovery tests invertibility.")
    else:
        emit("  (no sensitive-but-non-identifiable parameters at this threshold)")

    # Quantify the swap init_ec -> km2tgf if both extra bundles are given.
    if args.topk5_sobol and args.topk5_ident:
        s = load(args.topk5_sobol)
        d = load(args.topk5_ident)
        emit("")
        emit("=" * 70)
        emit("EFFECT OF THE FILTER: top-5-by-Sobol  vs  5-identifiable")
        emit("=" * 70)
        emit(f"  top-5 by Sobol      : {s['top_k']}")
        emit(f"    mean nRMSE        = {s['recovery']['nrmse_mean']:.3f}")
        emit(f"  5 identifiable      : {d['top_k']}")
        emit(f"    mean nRMSE        = {d['recovery']['nrmse_mean']:.3f}")
        improvement = (s['recovery']['nrmse_mean'] - d['recovery']['nrmse_mean'])
        pct = 100.0 * improvement / s['recovery']['nrmse_mean']
        emit("")
        emit(f"  swapping the non-identifiable init_ec for km2tgf lowers mean")
        emit(f"  nRMSE by {improvement:.3f} ({pct:.0f}% relative improvement).")
        emit("  -> Filtering on identifiability, not sensitivity alone, gives the")
        emit("     better-constrained calibration.")

    emit("")
    emit("=" * 70)

    out = Path(__file__).resolve().parent / "identifiability_report.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()