#!/usr/bin/env python3
"""Numbers of the forward benchmark from the result files: one-step metrics per model, cytokine,
seed and window (models/<model>_3d/res_*.json), the persistence and linear-extrapolation baselines
(models/persistence_baseline_run_0062_50.json) and the training and inference times. Prints
LaTeX rows for the tables of the paper and writes everything to models/forward_numbers.json.

usage:  python scripts/forward_numbers.py [--abm-runtime 4984.06]
"""
import json, argparse
from pathlib import Path
import numpy as np

MODELS = [("DeepONet", "deeponet_3d"), ("U-Net", "unet_3d")]
CYTS = [("il8", "IL-8"), ("il10", "IL-10")]
SEEDS = [1, 42, 100]
W = {"early": "Near_Horizon_t82_t91", "late": "Far_Horizon_t92_t100"}
PLANES = [("xy_midplane_z", "xy"), ("xz_midplane_y", "xz"), ("yz_midplane_x", "yz")]

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="models"); ap.add_argument("--run", default="run_0062"); ap.add_argument("--grid", type=int, default=50)
ap.add_argument("--abm-runtime", type=float, default=5121.0, help="wall-clock of one ABM trajectory (s)")
a = ap.parse_args()
M = Path(a.models)
res = {m: {c: {s: json.load(open(M / d / f"res_{c}_{a.run}_{a.grid}_{s}.json")) for s in SEEDS} for c, _ in CYTS} for m, d in MODELS}
base = json.load(open(M / f"persistence_baseline_{a.run}_{a.grid}.json"))
out = {"models": {}, "baselines": base}

def stat(m, c, w, key, sub=None):
    v = [res[m][c][s]["results"][W[w]][key] if sub is None else res[m][c][s]["results"][W[w]]["Slice_2D"][sub][key] for s in SEEDS]
    return float(np.mean(v)), float(np.min(v)), float(np.max(v)), [float(x) for x in v]

f3 = lambda x: f"{x:.3f}"
print("Table results_3d: global R2 (mean over seeds; range over seeds in brackets)")
for c, cl in CYTS:
    for m, _ in MODELS:
        e, l = stat(m, c, "early", "Global_R2"), stat(m, c, "late", "Global_R2")
        print(f"  {cl:6s} {m:9s} & ${f3(e[0])}$ (${f3(e[1])}$-${f3(e[2])}$) & ${f3(l[0])}$ (${f3(l[1])}$-${f3(l[2])}$) \\\\")
        out["models"].setdefault(m, {}).setdefault(c, {})["Global_R2"] = {"early": e[3], "late": l[3]}
    for b, bl in (("persistence", "Persistence"), ("linear_extrapolation", "Linear extrapolation")):
        print(f"  {cl:6s} {bl:20s} & ${f3(base[c][b][W['early']]['Global_R2'])}$ & ${f3(base[c][b][W['late']]['Global_R2'])}$ \\\\")

print("\nTable results_full: SSIM / Dice / Corr / masked RMSE, early window (mean over seeds); baselines in the same window")
for c, cl in CYTS:
    for m, _ in MODELS:
        vals = [stat(m, c, "early", k)[0] for k in ("SSIM", "Avg_Dice", "Spatial_Correlation", "Masked_RMSE")]
        out["models"][m][c]["early_metrics"] = dict(zip(("SSIM", "Dice", "Corr", "Masked_RMSE"), vals))
        print(f"  {cl:6s} {m:9s} & " + " & ".join(f"${f3(v)}$" for v in vals[:3]) + f" & ${vals[3]:.2e}$ \\\\")
    for b, bl in (("persistence", "Persistence"), ("linear_extrapolation", "Linear extrapolation")):
        r = base[c][b][W["early"]]
        print(f"  {cl:6s} {bl:20s} & " + " & ".join(f"${f3(r[k])}$" for k in ("SSIM", "Avg_Dice", "Spatial_Correlation")) + f" & ${r['Masked_RMSE']:.2e}$ \\\\")

print("\nTable seed_3d: global R2 per seed (early / late)")
for c, cl in CYTS:
    for m, _ in MODELS:
        cells = [f"${f3(res[m][c][s]['results'][W['early']]['Global_R2'])}$ / ${f3(res[m][c][s]['results'][W['late']]['Global_R2'])}$" for s in SEEDS]
        sd = np.std([res[m][c][s]["results"][W["early"]]["Global_R2"] for s in SEEDS])
        print(f"  {cl:6s} {m:9s} & " + " & ".join(cells) + f" \\\\   (s.d. over seeds, early: {sd:.3f})")

print("\nTable slice_3d: mid-plane R2 in the early window (mean over seeds)")
for c, cl in CYTS:
    for m, _ in MODELS:
        vals = [stat(m, c, "early", "R2", sub)[0] for sub, _ in PLANES]
        glob = stat(m, c, "early", "Global_R2")[0]
        out["models"][m][c]["midplane_R2_early"] = dict(zip([p for _, p in PLANES], vals))
        print(f"  {cl:6s} {m:9s} & " + " & ".join(f"${f3(v)}$" for v in vals) + f" & ${f3(glob)}$ \\\\")

print("\nTable cost_3d: training time (s, mean over seeds and cytokines), inference for 99 frames (s), ABM / inference")
for m, _ in MODELS:
    tr = np.mean([res[m][c][s]["train_time_seconds"] for c, _ in CYTS for s in SEEDS])
    pr = np.mean([res[m][c][s]["pred_time_seconds"] for c, _ in CYTS for s in SEEDS])
    out["models"][m]["train_time_s"] = float(tr); out["models"][m]["pred_time_s"] = float(pr); out["models"][m]["abm_over_inference"] = float(a.abm_runtime / pr)
    print(f"  {m:9s} & ${tr:.0f}$ & ${pr:.1f}$ & ${a.abm_runtime / pr:.0f}$ \\\\")
print(f"  ABM trajectory: {a.abm_runtime:.0f} s")

print("\nOptuna hyperparameters (seed 42)")
for m, _ in MODELS:
    for c, _ in CYTS:
        print(f"  {m:9s} {c:5s} {res[m][c][42]['best_params']}")
json.dump(out, open(M / "forward_numbers.json", "w"), indent=1)
print(f"\nsaved {M / 'forward_numbers.json'}")
