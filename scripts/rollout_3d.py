#!/usr/bin/env python3
"""Autoregressive rollout of the one-step field surrogates over the test window of the
benchmark run. From window --start on, the predicted cytokine channel of every frame is
fed back into the inputs of the following windows; the other five cytokine channels and
the cell masks still come from the ABM, because every model takes them as input. The
same metrics as in training are computed on the early and late windows, plus the global
R2 per step. The persistence rollout holds the last true frame before the start.

usage:  python scripts/rollout_3d.py --data preprocessed_3d --models models --out models/rollout_3d
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deeponet")); sys.path.insert(0, str(HERE / "unet"))
import train_deeponet_3d as DON
import train_unet_3d as UN

CYTS = ["il8", "il1", "il6", "il10", "tnf", "tgf"]


def load_model(kind, wdir, cyt, run, grid, seed):
    best = json.load(open(wdir / f"res_{cyt}_{run}_{grid}_{seed}.json"))["best_params"]
    w = wdir / f"weights_{cyt}_{run}_{grid}_{seed}.weights.h5"
    if kind == "deeponet":
        model = DON.DeepONet(hidden=best["hidden"], p=best["p"])
        model([np.zeros((1, 8), np.float32), np.zeros((1, 16, 25), np.float32)], training=False)
    else:
        model = UN.build_unet3d(grid_size=grid, in_channels=22, base_filters=best["base_filters"],
                                depth=best["depth"], dropout=best["dropout"])
    model.load_weights(str(w))
    return model


def predict_one(kind, model, xb, xt, idx):
    """xb (1, 2, G, G, G, 11), xt (1, G^3, 4) -> (G, G, G) prediction on the scaled axis."""
    G = xb.shape[2]
    if kind == "deeponet":
        return DON.predict_full(model, DON.build_branch_inputs(xb, xt, idx),
                                DON.build_trunk_inputs(xb, xt))[0, :, 0].reshape(G, G, G)
    X = xb.transpose(0, 2, 3, 4, 1, 5).reshape(1, G, G, G, 22)
    return model.predict(X, batch_size=1, verbose=0)[0, ..., 0]


def rollout(kind, model, Xb, Xt, idx, start):
    """Window i takes frames i and i+1 and predicts frame i+2; predicted frames replace
    the true ones in channel idx from window `start` on."""
    N = Xb.shape[0]; preds = {}
    out = np.zeros((N - start,) + Xb.shape[2:5], np.float32)
    for i in range(start, N):
        xb = Xb[i:i + 1].copy()
        for k, f in enumerate((i, i + 1)):
            if f in preds: xb[0, k, ..., idx] = preds[f]
        p = predict_one(kind, model, xb, Xt[i:i + 1], idx)
        preds[i + 2] = p; out[i - start] = p
    return out


def r2_per_step(yt, yp):
    yp = np.maximum(yp, 0.0)                                   # as in calculate_metrics
    return [float(1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-30)) for a, b in zip(yt, yp)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed_3d"); ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="models/rollout_3d"); ap.add_argument("--run", default="run_0062")
    ap.add_argument("--grid", type=int, default=50); ap.add_argument("--start", type=int, default=80)
    ap.add_argument("--kinds", nargs="+", default=["deeponet", "unet"])
    ap.add_argument("--cytokines", nargs="+", default=["il8", "il10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 100])
    a = ap.parse_args()
    dp = Path(a.data) / a.run / f"{a.grid}x{a.grid}x{a.grid}"
    Xb = np.load(dp / "X_branch.npy").astype(np.float32)
    Xt = np.load(dp / "X_trunk.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)
    M = np.load(dp / "Y_masks_spatial.npy").astype(np.float32)
    meta = json.load(open(dp / "metadata.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = a.start; windows = {"Near_Horizon_t82_t91": (80, 90), "Far_Horizon_t92_t100": (90, 99)}

    for cyt in a.cytokines:
        idx = CYTS.index(cyt); cmax = float(meta["scaling"]["max"][idx])
        Yt = DON.denormalize(Y[s:, ..., idx:idx + 1], cmax)                       # (N-s, G, G, G, 1)
        # persistence rollout: the last true frame before the start, repeated
        Yp_pers = DON.denormalize(np.repeat(Xb[s:s + 1, 1, ..., idx:idx + 1], Y.shape[0] - s, axis=0), cmax)
        res_pers = {k: DON.calculate_metrics(Yt[lo - s:hi - s], Yp_pers[lo - s:hi - s], M[lo:hi], cmax)
                    for k, (lo, hi) in windows.items()}
        res_pers["r2_per_step"] = r2_per_step(Yt, Yp_pers)
        json.dump({"cytokine": cyt, "start_window": s, "results": res_pers},
                  open(out / f"rollout_persistence_{cyt}_{a.run}_{a.grid}.json", "w"), indent=2)
        for kind in a.kinds:
            wdir = Path(a.models) / f"{kind}_3d"
            for seed in a.seeds:
                if not (wdir / f"weights_{cyt}_{a.run}_{a.grid}_{seed}.weights.h5").exists():
                    print(f"[skip] {kind} {cyt} seed {seed}: no weights"); continue
                t0 = time.time()
                model = load_model(kind, wdir, cyt, a.run, a.grid, seed)
                Yp = DON.denormalize(rollout(kind, model, Xb, Xt, idx, s)[..., None], cmax)
                res = {k: DON.calculate_metrics(Yt[lo - s:hi - s], Yp[lo - s:hi - s], M[lo:hi], cmax)
                       for k, (lo, hi) in windows.items()}
                res["r2_per_step"] = r2_per_step(Yt, Yp)
                json.dump({"model": kind, "cytokine": cyt, "seed": seed, "start_window": s,
                           "rollout_time_seconds": round(time.time() - t0, 1), "results": res},
                          open(out / f"rollout_{kind}_{cyt}_{a.run}_{a.grid}_{seed}.json", "w"), indent=2)
                print(f"{kind:9s} {cyt:5s} seed {seed:<4d} rollout R2 early {res['Near_Horizon_t82_t91']['Global_R2']:.3f} "
                      f"late {res['Far_Horizon_t92_t100']['Global_R2']:.3f} | persistence {res_pers['Near_Horizon_t82_t91']['Global_R2']:.3f} "
                      f"/ {res_pers['Far_Horizon_t92_t100']['Global_R2']:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
