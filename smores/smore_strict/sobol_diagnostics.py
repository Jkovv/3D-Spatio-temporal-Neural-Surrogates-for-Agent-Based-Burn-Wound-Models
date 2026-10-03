"""Diagnostics for the Sobol screen on volume-averaged replicate means
(sobol_repmean.py): (1) five-fold cross-validated R2 of the Gaussian-process
emulator of every observable; (2) total-order indices recomputed on nested
subsamples of 60, 70, 80, 90 and 100 sweep points (Saltelli base N = 512)."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smore"))
from observables import summarize_observable, FEATURE_NAMES, CYTOKINES
from sensitivity import emulator_sobol, _fit_gp
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser()
ap.add_argument("--replicates-npz", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n-saltelli", type=int, default=512)
a = ap.parse_args()
D = np.load(a.replicates_npz, allow_pickle=True)
runs, cyts = [str(r) for r in D["runs"]], [str(c) for c in D["cyts"]]
Y = np.nanmean(D["Y"], axis=1)[:, :, [cyts.index(c) for c in CYTOKINES]]
m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
pm = {r["run_id"]: r["params"] for r in m["runs"]}
theta = np.array([[pm[r][p] for p in names] for r in runs], float)
F = summarize_observable(Y); out = {"cv_r2": {}, "subsample": {}}

print("five-fold CV R2 of the emulator, per observable", flush=True)
kf = KFold(5, shuffle=True, random_state=0)
for j, fn in enumerate(FEATURE_NAMES):
    y = F[:, j]; yp = np.empty_like(y)
    for tr, te in kf.split(theta):
        pred, _ = _fit_gp(theta[tr], y[tr]); yp[te] = pred(theta[te])
    out["cv_r2"][fn] = float(1 - ((y - yp)**2).sum() / ((y - y.mean())**2).sum())
    print(f"  {fn:12s} {out['cv_r2'][fn]:+.3f}", flush=True)
for c in CYTOKINES:
    v = [out["cv_r2"][f] for f in FEATURE_NAMES if f.startswith(c + "_")]
    print(f"  {c:5s} mean {np.mean(v):+.3f}  min {np.min(v):+.3f}  max {np.max(v):+.3f}")

order = np.random.default_rng(0).permutation(len(runs))          # nested subsamples
for n in (60, 70, 80, 90, 100):
    idx = np.sort(order[:n])
    rk = emulator_sobol(theta[idx], F[idx], FEATURE_NAMES, names, bounds, n_saltelli=a.n_saltelli)["ranking"]
    out["subsample"][n] = {r["param"]: r["ST_mean"] for r in rk}
    print(f"N={n:3d}: " + "  ".join(f"{r['param']} {r['ST_mean']:.3f}" for r in rk[:6]), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
