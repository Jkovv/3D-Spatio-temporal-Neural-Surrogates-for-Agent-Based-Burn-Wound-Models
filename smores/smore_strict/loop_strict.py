"""Surrogate-in-the-loop with SMoRe ParS (IL-8).

Question: can volume-averaged trajectories predicted by the neural surrogate stand
in for ABM output as the data of the inference, once the surrogate's own error is
accounted for?  The library of bound surfaces always comes from the ABM arm; only
the held-out point's data box changes.  IL-8 is deterministic across realisations,
so in every arm the SM uncertainty is the fit residual (sigma mode 'paper'), and a
surrogate tolerance is added to it in quadrature, frame by frame.

Arms (all leave-one-out over the sweep points other than the surrogate's training run):
  abm         ABM trajectory (reference)
  sur         surrogate trajectory, no allowance for surrogate error (v1 experiment)
  sur_val     surrogate trajectory, tolerance = the surrogate's relative RMS error of the
              volume average on the validation frames of its own training run: what a
              user knows before applying the surrogate elsewhere
  sur_cv      surrogate trajectory, tolerance = per-frame relative RMS error of the
              surrogate over all OTHER sweep points (the held-out point is never used):
              an honest, out-of-distribution error model
              NOTE: this treats the error as independent noise in every frame.  The
              surrogate's error is correlated in time (a systematic offset of the whole
              trajectory), which a fit over 99 frames averages away, so this tolerance
              is too small by roughly sqrt(frames).  Kept to show exactly that.
  sur_conf    surrogate trajectory, error propagated where the inference happens: on the
              other sweep points the difference delta = phi(surrogate) - phi(ABM) between
              the SM parameters fitted to the two trajectories is computed (transformed
              scale), and the held-out data box is shifted and widened by its empirical
              quantiles (two-sided, Bonferroni over the SM parameters, split-conformal
              style).  Accounts for bias and temporal correlation; the held-out point's own
              ABM output is never used.  This is the main surrogate arm.
  abm_err_conf  control: ABM trajectory of the held-out point plus the surrogate's error
              trajectory from a different, randomly chosen sweep point, with the same
              conformal widening as sur_conf.  Same error size and temporal structure as
              the surrogate's, but unrelated to the held-out parameters.  If sur_conf
              recovers worse than abm_err_conf, the surrogate's errors are
              parameter-dependent: they distort the information the inference needs.
  abm_err_cv  the same control with the per-frame tolerance of sur_cv (optional arm).
"""
import copy
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smore_pars_strict as S

ap = argparse.ArgumentParser()
ap.add_argument("--trajectories-npz", required=True)     # from run_calibration_surrogate.py
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=8)
ap.add_argument("--n-cand", type=int, default=131072); ap.add_argument("--n-cand-max", type=int, default=4194304)
ap.add_argument("--n-min", type=int, default=300); ap.add_argument("--n-sd", type=float, default=2.0)
ap.add_argument("--params", nargs="+", default=None)
ap.add_argument("--exclude", nargs="*", default=["run_0062"])   # the surrogate's own training run
ap.add_argument("--train-run", default="run_0062"); ap.add_argument("--val-hours", nargs=2, type=float, default=[72, 81])
ap.add_argument("--arms", nargs="+", default=["abm", "sur", "sur_val", "sur_cv", "sur_conf", "abm_err_conf"])
ap.add_argument("--conf-alpha", type=float, default=0.05)   # joint level of the conformal widening (Bonferroni over SM parameters)
ap.add_argument("--folds", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
S.SIGMA_MODE = "paper"
D = np.load(a.trajectories_npz, allow_pickle=True)
runs_all = [str(r) for r in D["run_ids"]]; cyts = [str(c) for c in D["cytokines"]]
Ys_all, Ya_all = np.asarray(D["Y_sur"], float), np.asarray(D["Y_abm"], float)     # (n, T, C)
hours = np.asarray(D["t_grid"], float)/10000.0                                       # 10000 MCS = 1 h
keep = [i for i, r in enumerate(runs_all) if r not in set(a.exclude)]
runs, Ys, Ya = [runs_all[i] for i in keep], Ys_all[keep], Ya_all[keep]
m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
pm = {r["run_id"]: r["params"] for r in m["runs"]}
theta = np.array([[pm[r][p] for p in names] for r in runs], float)
n, T, C = Ya.shape; t = np.linspace(0, 1, T); zeros = np.zeros((T, C))
print(f"excluded from the loop: {a.exclude}; {n} points, {T} frames ({hours[0]:.0f}-{hours[-1]:.0f} h), cytokines {cyts}", flush=True)

def scale(Y): return np.maximum(np.abs(Y).max(axis=1, keepdims=True), 1e-300)          # (n, 1, C)
E = (Ys - Ya)/scale(Ya)                       # surrogate error relative to the run's ABM maximum, (n, T, C)
def tol_cv(k):                                 # per-frame RMS over all other points
    return np.sqrt(np.mean(np.delete(E, k, axis=0)**2, axis=0))                     # (T, C)
i0 = runs_all.index(a.train_run)
fr = np.where((hours >= a.val_hours[0]) & (hours <= a.val_hours[1]))[0]
e0 = (Ys_all[i0] - Ya_all[i0])/np.maximum(np.abs(Ya_all[i0]).max(axis=0, keepdims=True), 1e-300)
tol_val = np.sqrt(np.mean(e0[fr]**2, axis=0))                                       # (C,)
rng = np.random.default_rng(a.seed)
donor = np.array([rng.choice([j for j in range(n) if j != k]) for k in range(n)])
Y_err = Ya + E[donor]*scale(Ya)
rel_rms = lambda Y: np.sqrt(np.mean(((Y - Ya)/scale(Ya))**2, axis=1))[:, 0]
traj_r2 = lambda Y: np.array([1 - ((Ya[i, :, 0] - Y[i, :, 0])**2).sum()/((Ya[i, :, 0] - Ya[i, :, 0].mean())**2).sum() for i in range(n)])
print(f"  surrogate relative RMS error over the sweep: mean {rel_rms(Ys).mean():.4f}, median {np.median(rel_rms(Ys)):.4f}, max {rel_rms(Ys).max():.4f}; "
      f"trajectory R2 mean {traj_r2(Ys).mean():.3f}", flush=True)
print(f"  validation-frame error on {a.train_run} ({hours[fr[0]]:.0f}-{hours[fr[-1]]:.0f} h): {tol_val}; "
      f"cross-validated per-frame tolerance: median {np.median(np.sqrt(np.mean(E**2, axis=0))):.4f}", flush=True)
print(f"  control (transplanted error): relative RMS mean {rel_rms(Y_err).mean():.4f}, trajectory R2 mean {traj_r2(Y_err).mean():.3f}", flush=True)

ARM = {"abm": (Ya, None), "sur": (Ys, None), "sur_val": (Ys, "val"), "sur_cv": (Ys, "cv"), "abm_err_cv": (Y_err, "cv"),
       "sur_conf": (Ys, "conf"), "abm_err_conf": (Y_err, "conf")}
def fit_arm(Y, tol):
    if tol == "conf": tol = None                     # conformal widening is applied to the fitted boxes afterwards
    ex = [None if tol is None else (tol_val if tol == "val" else tol_cv(k)) for k in range(n)]
    jobs = [(Y[k], zeros, t, cyts, "paper", ex[k]) for k in range(n)]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as pool: return pool.map(S.fit_point, jobs)
    return [S.fit_point(j) for j in jobs]

fits_abm = fit_arm(Ya, None)
_cache = {}
def plain_fits(key, Y):                               # SM fits without any tolerance, computed once per data source
    if key not in _cache: _cache[key] = fit_arm(Y, None)
    return _cache[key]
keys_sm = S._sm_bound_matrix(fits_abm, runs)[0]
lob_sm = [S.SM[S.CYT_SM[nm]][4][j] for (nm, j, _) in keys_sm]
def best(fits): return np.array([[f[nm]["best"][j] for (nm, j, _) in keys_sm] for f in fits], float)
def tf_pair(x, y, lb):                                # one transform (and floor) for both sources
    if lb < 0: return x, y
    z = np.concatenate([x, y]); fl = 1e-3*np.median(z[z > 0])
    return np.log(np.maximum(x, fl)), np.log(np.maximum(y, fl))
def conformal(fits_data):
    """Shift and widen every held-out data box by the quantiles of delta = phi(sur) - phi(ABM) over the
    OTHER sweep points.  Box for phi(ABM): [lo - q_hi(delta), hi - q_lo(delta)] on the transformed scale."""
    Bs, Ba = best(plain_fits("sur", Ys)), best(fits_abm)
    delta = np.column_stack([np.subtract(*tf_pair(Bs[:, q], Ba[:, q], lob_sm[q])) for q in range(len(keys_sm))])
    a_q = a.conf_alpha/len(keys_sm); out = []; adj = []
    for k in range(n):
        d = np.delete(delta, k, axis=0)
        lo_adj = -np.quantile(d, 1 - a_q/2, axis=0, method="higher"); hi_adj = -np.quantile(d, a_q/2, axis=0, method="lower")
        f = copy.deepcopy(fits_data[k])
        for q, (nm, j, _) in enumerate(keys_sm):
            if lob_sm[q] < 0: f[nm]["lo"][j] += lo_adj[q]; f[nm]["hi"][j] += hi_adj[q]
            else: f[nm]["lo"][j] *= float(np.exp(lo_adj[q])); f[nm]["hi"][j] *= float(np.exp(hi_adj[q]))
        out.append(f); adj.append([lo_adj.tolist(), hi_adj.tolist()])
    med = np.median(np.abs(delta), axis=0)
    print(f"  conformal widening: median |delta| per SM parameter {dict(zip([c for _, _, c in keys_sm], np.round(med, 3)))} (transformed scale)", flush=True)
    return out, adj
sub = a.params or names; idx = [names.index(p) for p in sub]
folds = list(range(a.folds)) if a.folds else None
res = {"params": sub, "n_points": n, "excluded": a.exclude, "arms": {},
       "tolerance": {"val_rel_rms": tol_val.tolist(), "cv_rel_rms_median": float(np.median(np.sqrt(np.mean(E**2, axis=0))))},
       "surrogate_rel_rms": rel_rms(Ys).tolist(), "surrogate_traj_r2": traj_r2(Ys).tolist(),
       "control_rel_rms": rel_rms(Y_err).tolist(), "donor": [runs[j] for j in donor],
       "settings": {"n_cand": a.n_cand, "n_cand_max": a.n_cand_max, "n_min": a.n_min, "n_sd": a.n_sd, "sigma_mode": "paper",
                    "conf_alpha": a.conf_alpha}}
for arm in a.arms:
    Y, tol = ARM[arm]
    if arm == "abm": fits_data = fits_abm
    elif tol == "conf":
        fits_data, adj = conformal(plain_fits("sur" if arm == "sur_conf" else "err", Y))
        res.setdefault("conformal_adjust", {})[arm] = adj
    elif tol is None and arm == "sur": fits_data = plain_fits("sur", Ys)
    else: fits_data = fit_arm(Y, tol)
    w = [np.median((np.array(f["il8"]["hi"]) - np.array(f["il8"]["lo"]))/np.maximum(np.abs(f["il8"]["best"]), 1e-300)) for f in fits_data if f.get("il8")]
    print(f"\n[{arm}] data boxes: median relative interval width {np.median(w):.4f}", flush=True)
    X, acc, info = S.jain_region(theta[:, idx], fits_abm, runs, sub, bounds, n_cand=a.n_cand, procs=a.procs, n_sd=a.n_sd,
                                 data_fits=None if arm == "abm" else fits_data, n_min=a.n_min, n_cand_max=a.n_cand_max, folds=folds)
    J = S.summarise_region(X, acc, sub, info, n_sd=a.n_sd)
    S.print_region(J, sub, arm)
    J["data_fits"] = {r: f for r, f in zip(runs, fits_data)}
    res["arms"][arm] = J
    json.dump(res, open(a.out, "w"), indent=1, default=float)      # written after every arm
print(f"\nsaved {a.out}")
