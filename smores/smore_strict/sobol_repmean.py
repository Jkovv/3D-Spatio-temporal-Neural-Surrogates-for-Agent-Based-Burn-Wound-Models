"""Emulator-based Sobol screen on the same data as SMoRe ParS: volume-averaged
trajectories, averaged over the 6 realisations of every sweep point.
Two observable sets: (a) the four summaries per cytokine used before (final,
mean, max, AUC); (b) the fitted surrogate-model (ODE) parameters per cytokine."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smore"))
from observables import summarize_observable, FEATURE_NAMES, CYTOKINES
from sensitivity import emulator_sobol

ap = argparse.ArgumentParser()
ap.add_argument("--replicates-npz", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--fits-json", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-saltelli", type=int, default=1024)
a = ap.parse_args()
D = np.load(a.replicates_npz, allow_pickle=True)
runs, cyts = [str(r) for r in D["runs"]], [str(c) for c in D["cyts"]]
Ymean = np.nanmean(D["Y"], axis=1)                                   # (n, T, C)
Ymean = Ymean[:, :, [cyts.index(c) for c in CYTOKINES]]             # cytokine order of observables.py
m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
pm = {r["run_id"]: r["params"] for r in m["runs"]}
theta = np.array([[pm[r][p] for p in names] for r in runs], float)
out = {}
feats = summarize_observable(Ymean)
out["summaries"] = emulator_sobol(theta, feats, FEATURE_NAMES, names, bounds, n_saltelli=a.n_saltelli)
fits = json.load(open(a.fits_json))
sm_names, cols = [], []
for c in ["il8", "il6", "il10", "tnf", "tgf"]:
    f0 = fits[runs[0]][c]
    for j, pn in enumerate(f0["names"]):
        sm_names.append(f"{c}_{pn}")
        v = np.array([fits[r][c]["best"][j] for r in runs], float)
        cols.append(np.log(np.maximum(v, 1e-300)) if pn != "t0" else v)
out["sm_parameters"] = emulator_sobol(theta, np.column_stack(cols), sm_names, names, bounds, n_saltelli=a.n_saltelli)
json.dump(out, open(a.out, "w"), indent=1)
for key in ("summaries", "sm_parameters"):
    print(f"\nSobol ranking (mean total-order index) on {key}:")
    for r in out[key]["ranking"]: print(f"  {r['param']:10s} ST={r['ST_mean']:.3f}")
