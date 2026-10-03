"""SMoRe ParS (Jain 2022 / Bergman 2024) on the external E. coli growth-curve sweep
of Gong & Ying: 5 genomes x 29 media = 145 conditions, each measured in 6
replicate curves.  Replicates give the mean trajectory and its standard error,
exactly as the ABM replicates do; the surrogate model is the logistic growth
curve (M2); bounds by profile likelihood; the admissible region in the
9-dimensional input space is inferred for every held-out condition."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helpers"))
import smore_pars_strict as S
from validate_smore_external import load_sweep
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("--curves", required=True); ap.add_argument("--design", required=True); ap.add_argument("--media", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=8)
ap.add_argument("--n-cand", type=int, default=20000); ap.add_argument("--min-od", type=float, default=0.05)
a = ap.parse_args()
say = lambda msg: print(msg, flush=True)
theta, Y, _masks, t_grid, ids, names = load_sweep(a.curves, a.design, a.media, say)
theta, Y = np.asarray(theta, float), np.asarray(Y, float)
# group replicate curves by identical input vector (condition)
keys = [tuple(np.round(r, 9)) for r in theta]
cond = {}
for i, k in enumerate(keys): cond.setdefault(k, []).append(i)
conds = list(cond); print(f"{len(ids)} curves -> {len(conds)} conditions, replicates per condition: "
                           f"min {min(len(v) for v in cond.values())} max {max(len(v) for v in cond.values())}")
th = np.array([theta[cond[k][0]] for k in conds])
R = max(len(v) for v in cond.values()); T = Y.shape[1]
Yrep = np.full((len(conds), R, T), np.nan)
for ci, k in enumerate(conds):
    for r, i in enumerate(cond[k]): Yrep[ci, r] = Y[i]
nrep = np.sum(np.isfinite(Yrep).any(axis=2), axis=1)
mu = np.nanmean(Yrep, axis=1); se = np.nanstd(Yrep, axis=1, ddof=1)/np.sqrt(np.maximum(np.sum(np.isfinite(Yrep), axis=1), 1))
grow = np.nanmax(mu, axis=1) >= a.min_od
print(f"conditions with growth (max mean OD >= {a.min_od}): {grow.sum()} of {len(conds)}; the rest are excluded")
keep = np.where(grow)[0]; th, mu, se, nrep = th[keep], mu[keep], se[keep], nrep[keep]
runs = [f"cond_{i:03d}" for i in keep]
S.CYT_SM = {"od": "M2"}                                   # logistic growth as the surrogate model
tn = np.linspace(0, 1, T)
def one(i):
    ok = np.isfinite(mu[i])
    return S.fit_point((mu[i][ok][:, None], se[i][ok][:, None], tn[ok], ["od"]))
with Pool(a.procs) as pool: fits = pool.map(one, range(len(runs)))
r2 = [f["od"]["r2"] for f in fits if f.get("od")]
print(f"logistic SM fit on replicate means: median R2 {np.median(r2):.3f}, min {np.min(r2):.3f}, "
      f"below 0.9: {sum(v < 0.9 for v in r2)} of {len(r2)}", flush=True)
bounds = {p: {"low": float(th[:, j].min()), "high": float(th[:, j].max())} for j, p in enumerate(names)}
C, X, acc = S.jain_region(th, fits, runs, names, bounds, n_cand=a.n_cand, procs=a.procs)
res = S.summarise_region(C, X, acc, names, th)
print(f"[Jain 2022 region] conditions with a non-empty region: {100*(1-res['frac_empty']):.0f}%")
print(f"{'input':14s} {'R2':>7s} {'cov95':>6s} {'sd ratio':>9s}")
for p in names:
    q = res["per_param"][p]; print(f"{p:14s} {q['r2_median_accepted']:7.3f} {q['coverage95']:6.2f} {q['sd_ratio']:9.2f}")
res["n_conditions"] = int(len(runs)); res["replicates_per_condition"] = nrep.tolist(); res["sm_r2"] = r2
res["fits"] = {r: f for r, f in zip(runs, fits)}
json.dump(res, open(a.out, "w"), indent=1, default=float)
