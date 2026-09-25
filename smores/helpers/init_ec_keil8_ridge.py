#!/usr/bin/env python3

import argparse, json, os, sys, csv
from pathlib import Path
import numpy as np


def il8_observable_from_file(mc_path, which):
    t, il8 = [], []
    with open(mc_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row:
                continue
            try:
                t.append(float(row[0])); il8.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    il8 = np.asarray(il8, float); t = np.asarray(t, float)
    if il8.size == 0:
        return np.nan
    if which == "final": return il8[-1]
    if which == "mean":  return float(np.mean(il8))
    if which == "max":   return float(np.max(il8))
    if which == "auc":   return float(np.trapz(il8, t)) if t.size == il8.size else float(np.sum(il8))
    raise ValueError(which)


def find_mean_conc(run_out_dir):
    for root, _d, files in os.walk(run_out_dir):
        if "mean_concentration.txt" in files:
            return os.path.join(root, "mean_concentration.txt")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--observable", default="il8_final")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--out", default="ridge_init_ec_keil8.json")
    args = ap.parse_args()

    which = args.observable.split("_", 1)[1] if "_" in args.observable else "final"

    man = json.load(open(args.manifest))
    names = man["param_names"]
    bounds = man["bounds"]
    baselines = man["baselines"]
    runs = man["runs"]
    i_ec = names.index("init_ec")
    i_k8 = names.index("keil8")

    sim_root = Path(args.sim_root)
    theta_rows, y_rows = [], []
    for run in runs:
        mc = find_mean_conc(sim_root / run["run_id"])
        if mc is None:
            print(f"  [skip] no mean_concentration.txt for {run['run_id']}")
            continue
        y = il8_observable_from_file(mc, which)
        if not np.isfinite(y):
            continue
        theta_rows.append([run["params"][n] for n in names])
        y_rows.append(y)
    theta = np.asarray(theta_rows, float)
    y = np.asarray(y_rows, float)
    if len(y) < 10:
        sys.exit(f"ERROR: only {len(y)} usable runs found under {sim_root}")
    print(f"  training emulator on {len(y)} runs, observable = IL8_{which}")

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    xs = StandardScaler().fit(theta)
    ys = StandardScaler().fit(y.reshape(-1, 1))
    Xn = xs.transform(theta)
    yn = ys.transform(y.reshape(-1, 1)).ravel()
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(theta.shape[1]), nu=2.5)
              + WhiteKernel(1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=4, random_state=0)
    gp.fit(Xn, yn)
    cv = cross_val_score(gp, Xn, yn, cv=5, scoring="r2")
    print(f"  emulator 5-fold CV R^2 = {cv.mean():.3f} (+/- {cv.std():.3f})")

    base = np.array([baselines[n] for n in names], float)
    ec_lo, ec_hi = bounds["init_ec"]["low"], bounds["init_ec"]["high"]
    k8_lo, k8_hi = bounds["keil8"]["low"], bounds["keil8"]["high"]
    ec_grid = np.linspace(ec_lo, ec_hi, args.grid)
    k8_grid = np.linspace(k8_lo, k8_hi, args.grid)

    Z = np.zeros((args.grid, args.grid))
    for a, ec in enumerate(ec_grid):
        row = np.tile(base, (args.grid, 1))
        row[:, i_ec] = ec
        row[:, i_k8] = k8_grid
        Z[a, :] = ys.inverse_transform(
            gp.predict(xs.transform(row)).reshape(-1, 1)).ravel()

    EC, K8 = np.meshgrid(ec_grid, k8_grid, indexing="ij")
    ec_n = (EC - ec_lo) / (ec_hi - ec_lo)
    k8_n = (K8 - k8_lo) / (k8_hi - k8_lo)
    prod = ec_n * k8_n
    c_prod = np.corrcoef(prod.ravel(), Z.ravel())[0, 1]
    c_ec = np.corrcoef(EC.ravel(), Z.ravel())[0, 1]
    c_k8 = np.corrcoef(K8.ravel(), Z.ravel())[0, 1]

    nb = 12
    edges = np.linspace(prod.min(), prod.max(), nb + 1)
    within = []
    for b in range(nb):
        m = (prod >= edges[b]) & (prod < edges[b + 1])
        if m.sum() > 3:
            within.append(Z[m].std())
    along_ridge_std = float(np.mean(within)) if within else float("nan")
    total_std = float(Z.std())
    flatness = along_ridge_std / total_std if total_std > 0 else float("nan")

    out = {
        "observable": args.observable,
        "n_train": int(len(y)),
        "emulator_cv_r2_mean": float(cv.mean()),
        "emulator_cv_r2_std": float(cv.std()),
        "corr_obs_product": float(c_prod),
        "corr_obs_init_ec": float(c_ec),
        "corr_obs_keil8": float(c_k8),
        "along_ridge_std": along_ridge_std,
        "total_std": total_std,
        "flatness_ratio": flatness,
    }
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"  {'quantity':32s} {'value':>10s}")
    print(f"  {'observable':32s} {args.observable:>10s}")
    print(f"  {'runs used':32s} {len(y):>10d}")
    print(f"  {'emulator CV R^2':32s} {cv.mean():>10.3f}")
    print(f"  {'|corr(obs, init_ec x keil8)|':32s} {abs(c_prod):>10.3f}")
    print(f"  {'|corr(obs, init_ec)|':32s} {abs(c_ec):>10.3f}")
    print(f"  {'|corr(obs, keil8)|':32s} {abs(c_k8):>10.3f}")
    print(f"  {'flatness (along/total std)':32s} {flatness:>10.3f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()