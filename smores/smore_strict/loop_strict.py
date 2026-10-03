"""Surrogate-in-the-loop with SMoRe ParS: the same IL-8 surrogate model, profile-
likelihood bounds and admissible-region inference as for the ABM, but with the
held-out point's data box fitted to the trajectory PREDICTED BY THE NEURAL
SURROGATE (volume-averaged, same preprocessing in both arms).  The library of
bound surfaces always comes from the ABM arm.  IL-8 is deterministic across ABM
realisations, so the SM uncertainty is the fit residual in both arms."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smore_pars_strict as S

ap = argparse.ArgumentParser()
ap.add_argument("--trajectories-npz", required=True)     # from run_calibration_surrogate.py
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=8); ap.add_argument("--n-cand", type=int, default=20000)
ap.add_argument("--params", nargs="+", default=None)
a = ap.parse_args()
D = np.load(a.trajectories_npz, allow_pickle=True)
runs = [str(r) for r in D["run_ids"]]; cyts = [str(c) for c in D["cytokines"]]
Y_sur, Y_abm = D["Y_sur"], D["Y_abm"]                     # (n, T, C) in the loop's cytokine order
m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
pm = {r["run_id"]: r["params"] for r in m["runs"]}
theta = np.array([[pm[r][p] for p in names] for r in runs], float)
T = Y_abm.shape[1]; t = np.linspace(0, 1, T); zeros = np.zeros((T, len(cyts)))
print(f"{len(runs)} points, {T} frames, cytokines {cyts}", flush=True)
from multiprocessing import Pool
with Pool(a.procs) as pool:
    fits_abm = pool.map(S.fit_point, [(Y_abm[i], zeros, t, cyts) for i in range(len(runs))])
    fits_sur = pool.map(S.fit_point, [(Y_sur[i], zeros, t, cyts) for i in range(len(runs))])
for lab, fits in (("ABM", fits_abm), ("surrogate", fits_sur)):
    for c in cyts:
        if c in S.CYT_SM:
            r2 = [f[c]["r2"] for f in fits if f.get(c)]
            print(f"  SM fit on {lab:9s} {c}: median R2 {np.median(r2):.3f}, min {np.min(r2):.3f}", flush=True)
sub = a.params or names; idx = [names.index(p) for p in sub]
res = {"params": sub, "n_points": len(runs)}
for arm, data_fits in (("abm", None), ("surrogate", fits_sur)):
    print(f"\n[{arm} arm] admissible regions ...", flush=True)
    C, X, acc = S.jain_region(theta[:, idx], fits_abm, runs, sub, bounds, n_cand=a.n_cand, procs=a.procs, data_fits=data_fits)
    res[arm] = S.summarise_region(C, X, acc, sub, theta[:, idx])
    J = res[arm]
    print(f"  folds with a non-empty region: {100*(1-J['frac_empty']):.0f}%")
    print(f"  {'param':9s} {'R2':>7s} {'cov95':>6s} {'sd ratio':>9s}")
    for p in sub:
        q = J["per_param"][p]; print(f"  {p:9s} {q['r2_median_accepted']:7.3f} {q['coverage95']:6.2f} {q['sd_ratio']:9.2f}")
    if "ridge" in J: print("  ridge:", {k: round(v, 3) for k, v in J["ridge"].items()})
res["fits_abm"] = {r: f for r, f in zip(runs, fits_abm)}; res["fits_surrogate"] = {r: f for r, f in zip(runs, fits_sur)}
json.dump(res, open(a.out, "w"), indent=1, default=float)
