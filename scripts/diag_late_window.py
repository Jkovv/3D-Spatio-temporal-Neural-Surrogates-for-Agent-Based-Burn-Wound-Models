#!/usr/bin/env python3
"""Where does a one-step prediction go wrong? Per test frame and seed: global R2, R2 over the
active voxels (truth > 0) and over the background, mean and maximum of the prediction relative
to the truth, and the share of negative predictions. Written for the IL-10 DeepONets, usable
for any model/cytokine.

usage:  python scripts/diag_late_window.py --kind deeponet --cytokine il10 --seeds 1 42 100
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_3d as R
DON = R.DON


def r2(a, b):
    sst = ((a - a.mean()) ** 2).sum()
    return float(1 - ((a - b) ** 2).sum() / sst) if sst > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed_3d"); ap.add_argument("--models", default="models")
    ap.add_argument("--run", default="run_0062"); ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--kind", default="deeponet"); ap.add_argument("--cytokine", default="il10")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 100])
    ap.add_argument("--frames", nargs=2, type=int, default=[80, 99], help="window indices (start, stop)")
    a = ap.parse_args()
    dp = Path(a.data) / a.run / f"{a.grid}x{a.grid}x{a.grid}"
    Xb = np.load(dp / "X_branch.npy").astype(np.float32); Xt = np.load(dp / "X_trunk.npy").astype(np.float32)
    Y = np.load(dp / "Y_target.npy").astype(np.float32)
    meta = json.load(open(dp / "metadata.json"))
    idx = R.CYTS.index(a.cytokine); cmax = float(meta["scaling"]["max"][idx])
    lo, hi = a.frames
    truth = DON.denormalize(Y[lo:hi, ..., idx], cmax)
    tr_max = DON.denormalize(Y[:70, ..., idx], cmax).max()
    print(f"{a.kind} {a.cytokine}: training-frame maximum {tr_max:.3e}; test frames {lo}-{hi - 1} (t = {lo + 2}-{hi + 1} h)")
    print(f"truth per frame: max/train-max " + " ".join(f"{truth[i].max() / tr_max:.2f}" for i in range(truth.shape[0])))
    for seed in a.seeds:
        wdir = Path(a.models) / f"{a.kind}_3d"
        model = R.load_model(a.kind, wdir, a.cytokine, a.run, a.grid, seed)
        if a.kind == "deeponet":
            p = DON.predict_full(model, DON.build_branch_inputs(Xb[lo:hi], Xt[lo:hi], idx), DON.build_trunk_inputs(Xb[lo:hi], Xt[lo:hi]))
            pred = DON.denormalize(p[:, :, 0].reshape(-1, a.grid, a.grid, a.grid), cmax)
        else:
            X = Xb[lo:hi].transpose(0, 2, 3, 4, 1, 5).reshape(-1, a.grid, a.grid, a.grid, 22)
            pred = DON.denormalize(model.predict(X, batch_size=1, verbose=0)[..., 0], cmax)
        print(f"\nseed {seed}")
        print(f"  {'frame':>5s} {'t(h)':>4s} {'R2':>7s} {'R2 active':>9s} {'R2 bkg':>8s} {'mean p/t':>8s} {'max p/t':>7s} {'neg %':>6s} {'bkg |err|/max':>13s}")
        for i in range(truth.shape[0]):
            t, pcl = truth[i], np.maximum(pred[i], 0.0)
            act = t > 0; bkg = ~act
            print(f"  {lo + i:5d} {lo + i + 2:4d} {r2(t, pcl):7.3f} {r2(t[act], pcl[act]):9.3f} {r2(t[bkg], pcl[bkg]) if bkg.any() else float('nan'):8.3f} "
                  f"{pcl.mean() / max(t.mean(), 1e-300):8.3f} {pcl.max() / max(t.max(), 1e-300):7.3f} {100 * np.mean(pred[i] < 0):6.1f} "
                  f"{np.abs(pcl[bkg]).mean() / max(t.max(), 1e-300) if bkg.any() else float('nan'):13.4f}")


if __name__ == "__main__":
    main()
