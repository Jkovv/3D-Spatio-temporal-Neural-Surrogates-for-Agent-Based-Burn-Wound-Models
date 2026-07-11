#!/usr/bin/env python3
"""
helpers/compute_speedup_report.py

Turns the surrogate timing JSONs and the measured ABM wall-clock into the
speed-up / break-even table for the paper (tab:cost_3d). Every number is read
from files produced by this pipeline; nothing is carried over from the thesis.

    speed-up = T_ABM / T_infer
    N*       = T_train / (T_ABM - T_infer)

T_ABM  : one 50^3 ABM run on rome  (from sweep .../abm_walltime.txt, or --t-abm)
T_train: surrogate training time   (train_time_seconds in each res_*.json)
T_infer: surrogate inference time  (pred_time_seconds  in each res_*.json)

Training and inference are averaged over the three seeds (1/42/100), reported
as mean +/- SD, matching the thesis convention.

USAGE
-----
    python helpers/compute_speedup_report.py \
        --models-root ../models \
        --t-abm 4984.06

(or --abm-walltime path to a sweep abm_walltime.txt to read T_ABM from it)
"""

import argparse
import json
import re
import statistics as st
from pathlib import Path
from collections import defaultdict


def read_t_abm(path):
    with open(path) as f:
        for line in f:
            if line.startswith("T_ABM_seconds"):
                return float(line.split("=")[1])
    raise SystemExit(f"no T_ABM_seconds in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="../models",
                    help="Root containing unet_3d/ and deeponet_3d/ with res_*.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--t-abm", type=float, help="ABM run time in seconds.")
    g.add_argument("--abm-walltime", help="Path to abm_walltime.txt.")
    args = ap.parse_args()

    t_abm = args.t_abm if args.t_abm else read_t_abm(args.abm_walltime)
    root = Path(args.models_root)

    # collect (model, cytokine) -> list of (train, infer) over seeds
    buckets = defaultdict(list)
    pat = re.compile(r"res_(\w+?)_run_\d+_\d+_(\d+)\.json$")
    model_dirs = {"U-Net": root / "unet_3d", "DeepONet": root / "deeponet_3d"}
    for model, d in model_dirs.items():
        if not d.exists():
            print(f"[warn] {d} not found, skipping {model}")
            continue
        for jf in sorted(d.glob("res_*.json")):
            m = pat.search(jf.name)
            if not m:
                continue
            cyt = m.group(1)
            with open(jf) as f:
                j = json.load(f)
            tr = j.get("train_time_seconds")
            inf = j.get("pred_time_seconds")
            if tr is None or inf is None:
                print(f"[warn] {jf.name} missing timing, skipping")
                continue
            buckets[(model, cyt)].append((tr, inf))

    if not buckets:
        raise SystemExit("no timing JSONs found")

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 74)
    emit("SURROGATE SPEED-UP / BREAK-EVEN  (tab:cost_3d)")
    emit("=" * 74)
    emit(f"T_ABM = {t_abm:.0f} s  (one 50^3 ABM run, rome/CPU)")
    emit("training + inference averaged over seeds 1/42/100 (mean +/- SD)")
    emit("")
    emit(f"{'model':9} {'cyt':5} {'n':>2} {'T_train (s)':>16} "
         f"{'T_infer (s)':>15} {'speed-up':>9} {'N*':>7}")
    emit("-" * 70)

    for (model, cyt) in sorted(buckets):
        vals = buckets[(model, cyt)]
        tr = [v[0] for v in vals]
        inf = [v[1] for v in vals]
        tr_m = st.mean(tr); tr_s = st.pstdev(tr) if len(tr) > 1 else 0.0
        if_m = st.mean(inf); if_s = st.pstdev(inf) if len(inf) > 1 else 0.0
        speedup = t_abm / if_m
        nstar = tr_m / (t_abm - if_m) if t_abm > if_m else float("inf")
        emit(f"{model:9} {cyt:5} {len(vals):>2} "
             f"{tr_m:8.1f} +/- {tr_s:4.1f}  {if_m:7.2f} +/- {if_s:4.2f}  "
             f"{speedup:8.0f}x {nstar:6.2f}")

    emit("")
    emit("speed-up = T_ABM / T_infer   (compute saved per surrogate query)")
    emit("N*       = T_train / (T_ABM - T_infer)   (runs to amortise training)")
    emit("N* < 1 means the surrogate pays for its own training within one ABM run.")
    emit("=" * 74)

    out = Path(__file__).resolve().parent / "speedup_report.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()