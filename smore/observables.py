import json
import csv
from pathlib import Path

import numpy as np

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
_MEAN_COLS = [f"{c}mean" for c in CYTOKINES]

def _read_mean_concentration(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    idx_time = header.index("meanconcen")
    idx_mean = [header.index(c) for c in _MEAN_COLS]
    mcs, means = [], []
    for r in rows[1:]:
        if not r:
            continue
        mcs.append(int(float(r[idx_time])))
        means.append([float(r[i]) for i in idx_mean])
    order = np.argsort(mcs)
    return np.array(mcs)[order], np.array(means)[order]


def _read_theta(run_dir, param_names):
    p = run_dir / "params.json"
    if not p.exists():
        p = run_dir / "resolved_params.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{run_dir}: no params.json/resolved_params.json -> cannot pair "
            f"theta with trajectory (this is exactly the desync bug we fixed).")
    doc = json.load(open(p))
    params = doc.get("params", doc.get("overrides_from_json", doc))
    missing = [n for n in param_names if n not in params]
    if missing:
        raise KeyError(f"{run_dir}: params.json missing {missing}")
    return np.array([float(params[n]) for n in param_names])

def _resample(mcs, series, n_points):
    if len(mcs) == n_points and np.all(np.diff(mcs) > 0):
        grid = np.linspace(mcs[0], mcs[-1], n_points)
        if np.allclose(grid, mcs):
            return series
    grid = np.linspace(mcs[0], mcs[-1], n_points)
    out = np.empty((n_points, series.shape[1]))
    for k in range(series.shape[1]):
        out[:, k] = np.interp(grid, mcs, series[:, k])
    return out


def load_sweep(sim_root, param_names, n_time=None):
    sim_root = Path(sim_root)
    run_dirs = sorted(d for d in sim_root.iterdir()
                      if d.is_dir() and d.name.startswith("run_"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* dirs under {sim_root}")

    raw = []
    for rd in run_dirs:
        mc_path = rd / "datafiles"/"mean_concentration.txt"
        if not mc_path.exists():
            print(f"[observables] WARNING {rd.name}: no mean_concentration.txt, "
                  f"skipping")
            continue
        theta = _read_theta(rd, param_names)
        mcs, means = _read_mean_concentration(mc_path)
        raw.append((rd.name, theta, mcs, means))

    if not raw:
        raise RuntimeError(f"No usable runs under {sim_root}")

    # Common time length = min across runs unless overridden.
    T = n_time or min(len(m) for _, _, m, _ in raw)
    run_ids, thetas, Ys = [], [], []
    t_grid = None
    for name, theta, mcs, means in raw:
        res = _resample(mcs, means, T)
        if t_grid is None:
            t_grid = np.linspace(mcs[0], mcs[-1], T)
        run_ids.append(name)
        thetas.append(theta)
        Ys.append(res)

    return (np.array(thetas), np.array(Ys), run_ids, t_grid)


def summarize_observable(Y):
    n, T, C = Y.shape
    feats = []
    for c in range(C):
        s = Y[:, :, c]
        final = s[:, -1]
        mean = s.mean(axis=1)
        mx = s.max(axis=1)
        _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = _trap(s, axis=1) / max(1, T - 1)
        feats.append(np.stack([final, mean, mx, auc], axis=1))
    return np.concatenate(feats, axis=1)

FEATURE_NAMES = [f"{c}_{stat}" for c in CYTOKINES
                 for stat in ("final", "mean", "max", "auc")]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    names = man["param_names"]
    theta, Y, ids, t = load_sweep(args.sim_root, names)
    print(f"loaded {len(ids)} runs | theta {theta.shape} | Y {Y.shape}")
    print(f"feature matrix {summarize_observable(Y).shape}")