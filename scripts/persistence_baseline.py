#!/usr/bin/env python3
"""Trivial one-step baselines for the neural field surrogates (benchmark run run_0062, 50^3).

Every surrogate prediction of frame t is conditioned on the true ABM frames t-1 and t-2, so the
neural models are compared with two predictors that use the same information and nothing else:
  persistence           y_hat(t) = y(t-1)
  linear extrapolation  y_hat(t) = max(2 y(t-1) - y(t-2), 0)
The metrics are computed with calculate_metrics() of the U-Net training script, on the same
denormalised fields and the same windows (early: samples 80-89 = t82-91 h; late: 90-98 = t92-100 h),
so the numbers are directly comparable with models/*/res_*.json.

Run from the repository root:  python scripts/persistence_baseline.py
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "unet"))
from train_unet_3d import calculate_metrics, denormalize   # identical metric definitions

RUN, GRID = "run_0062", 50
CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
WINDOWS = {"Near_Horizon_t82_t91": slice(80, 90), "Far_Horizon_t92_t100": slice(90, 99)}

data = Path(f"preprocessed_3d/{RUN}/{GRID}x{GRID}x{GRID}")
meta = json.load(open(data / "metadata.json"))
M = np.load(data / "Y_masks_spatial.npy").astype(np.float32)
out = {}
for cyt in ("il8", "il10"):
    idx = CYTS.index(cyt); cmax = float(meta["scaling"]["max"][idx])
    Y = denormalize(np.load(data / "Y_target.npy").astype(np.float32)[..., idx:idx+1], cmax)
    pers = np.empty_like(Y); lin = np.empty_like(Y)
    pers[1:] = Y[:-1]; pers[0] = Y[0]
    lin[2:] = np.maximum(2*Y[1:-1] - Y[:-2], 0.0); lin[:2] = pers[:2]
    out[cyt] = {}
    for name, pred in (("persistence", pers), ("linear_extrapolation", lin)):
        out[cyt][name] = {w: calculate_metrics(Y[s], pred[s], M[s], cmax) for w, s in WINDOWS.items()}
    for name in ("persistence", "linear_extrapolation"):
        r = out[cyt][name]
        print(f"{cyt:5s} {name:21s} " + "  ".join(
            f"{w.split('_')[0]}: R2 {r[w]['Global_R2']:.4f} SSIM {r[w]['SSIM']:.4f} Dice {r[w]['Avg_Dice']:.4f} "
            f"Corr {r[w]['Spatial_Correlation']:.4f}" for w in WINDOWS), flush=True)
Path("models").mkdir(exist_ok=True)
json.dump(out, open("models/persistence_baseline_run_0062_50.json", "w"), indent=2, default=float)
print("saved models/persistence_baseline_run_0062_50.json")
