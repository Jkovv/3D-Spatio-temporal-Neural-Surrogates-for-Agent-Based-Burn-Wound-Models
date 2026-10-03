"""
SMoRe ParS following Jain et al. (2022) and Bergman et al. (2024).

Stage 1  explicit surrogate model (SM) per cytokine, chosen by AIC on the
         volume-averaged ABM trajectories (sm_selection.py):
           IL-8                : M3  c0 e^{-kt} + p/(k-lam) (e^{-lam t} - e^{-kt})
           IL-6, IL-10, TNF-a  : M1  c0 e^{-kt} + p/k (1 - e^{-kt})
           TGF-b               : M2  A / (1 + e^{-r (t - t0)})
           IL-1b excluded (no candidate reaches R2 > 0.9 on most runs).
Stage 2  at every sweep point: SM fitted to the replicate-mean trajectory by
         weighted least squares (weights = standard error across replicates,
         Jain eq. 2); 95% bounds on every SM parameter by profile likelihood
         (Delta chi2 = 3.84).  Each sweep point thus gives a box in SM space.
Stage 3  leave-one-out: the held-out point's own SM bounds play the role of
         the data-derived set Phi (Bergman 2024); a sweep point is accepted iff
         its box intersects Phi in every SM parameter.  Direct method (Bergman
         2024): the same number of points with the lowest weighted RSS between
         trajectories.
"""
import numpy as np, json, sys, argparse, os
from scipy.optimize import curve_fit
from multiprocessing import Pool

def m1(t, c0, p, k): return c0*np.exp(-k*t) + p/k*(1-np.exp(-k*t))
def m2(t, A, r, t0): return A/(1+np.exp(-r*(t-t0)))
def m3(t, c0, p, k, lam):
    d = k - lam; d = np.where(np.abs(d) < 1e-6, 1e-6, d)
    return c0*np.exp(-k*t) + p/d*(np.exp(-lam*t) - np.exp(-k*t))

# name -> (function, parameter names, amplitude flags, p0 list, lower, upper)
SM = {
 "M1": (m1, ["c0", "p", "k"], [1, 1, 0], [[0.1, 1, 1], [0.5, 5, 5], [0.0, 20, 20]], [0, 0, 1e-3], [2, 1e3, 1e3]),
 "M2": (m2, ["A", "r", "t0"], [1, 0, 0], [[1, 8, 0.5], [1, 3, 0.2], [1, 20, 0.8]], [0, 0, -1], [10, 200, 2]),
 "M3": (m3, ["c0", "p", "k", "lam"], [1, 1, 0, 0], [[0.1, 5, 5, 1], [0.5, 20, 2, 0.5], [0.0, 50, 10, 3]],
        [0, 0, 1e-3, 0], [2, 1e4, 1e3, 1e3]),
}
CYT_SM = {"il8": "M3", "il6": "M1", "il10": "M1", "tnf": "M1", "tgf": "M2"}
CHI2_95 = 3.841

def wfit(f, t, z, sig, p0s, lo, hi, fixed=None):
    """Weighted LS fit; `fixed` = (index, value) holds one parameter fixed."""
    n = len(p0s[0])
    free = [i for i in range(n) if fixed is None or i != fixed[0]]
    def g(tt, *q):
        full = np.empty(n); full[free] = q
        if fixed is not None: full[fixed[0]] = fixed[1]
        return f(tt, *full)
    best = (np.inf, None)
    for p0 in p0s:
        q0 = np.clip([p0[i] for i in free], [lo[i] for i in free], [hi[i] for i in free])
        try:
            q, _ = curve_fit(g, t, z, p0=q0, sigma=sig, absolute_sigma=True,
                             bounds=([lo[i] for i in free], [hi[i] for i in free]), maxfev=20000)
            chi2 = float((((z - g(t, *q))/sig)**2).sum())
            if chi2 < best[0]:
                full = np.empty(n); full[free] = q
                if fixed is not None: full[fixed[0]] = fixed[1]
                best = (chi2, full)
        except Exception:
            pass
    return best

def profile_bounds(f, t, z, sig, p_best, chi2_min, j, lo, hi, p0s):
    """95% profile-likelihood bounds for parameter j: step outwards from the
    optimum until chi2 exceeds chi2_min + 3.84 (or the parameter bound is hit),
    then bisect for the crossing point."""
    b = p_best[j]; thr = chi2_min + CHI2_95
    def chi2_at(v): return wfit(f, t, z, sig, [list(p_best)], lo, hi, fixed=(j, v))[0]
    log = lo[j] >= 0 and b > 0
    def move(v, d, sign):                      # d = step size (relative if log)
        return v*np.exp(sign*d) if log else v + sign*d*(hi[j]-lo[j])
    out = []
    for sign in (-1, +1):
        inside, d, v = b, 0.01, None
        while True:
            v = move(inside, d, sign)
            if (sign < 0 and v <= lo[j]) or (sign > 0 and v >= hi[j]):
                v = lo[j] if sign < 0 else hi[j]
                if chi2_at(v) <= thr: out.append(v); break
            if chi2_at(v) > thr:
                a, c = inside, v                    # bisection between inside and outside
                for _ in range(10):
                    m = np.sqrt(a*c) if (log and a > 0 and c > 0) else 0.5*(a + c)
                    if chi2_at(m) <= thr: a = m
                    else: c = m
                out.append(a); break
            inside, d = v, d*2
            if d > 20: out.append(lo[j] if sign < 0 else hi[j]); break
    return float(min(out)), float(max(out))

def fit_point(args):
    """One sweep point: returns per cytokine best params and 95% bounds (physical units)."""
    mu, se, t, cyts = args
    out = {}
    for c, name in enumerate(cyts):
        if name not in CYT_SM: continue
        f, pn, amp, p0s, lo, hi = SM[CYT_SM[name]]
        y = mu[:, c]; s = max(np.abs(y).max(), 1e-30)
        z = y/s
        sig = se[:, c]/s
        if np.median(sig) <= 1e-12:
            # deterministic output across replicates (volume-averaged IL-8): the only
            # uncertainty left is the surrogate-model approximation error, taken as the
            # RMS residual of an unweighted fit
            _, pu = wfit(f, t, z, np.ones_like(z), p0s, lo, hi)
            rms = np.sqrt(np.mean((z - f(t, *pu))**2)) if pu is not None else 1e-3
            sig = np.full_like(z, max(rms, 1e-6))
        else:
            floor = max(0.05*np.median(sig[sig > 0]), 1e-6)
            sig = np.maximum(sig, floor)
        chi2, pb = wfit(f, t, z, sig, p0s, lo, hi)
        if pb is None:
            out[name] = None; continue
        rng = [profile_bounds(f, t, z, sig, pb, chi2, j, lo, hi, p0s) for j in range(len(pn))]
        scale = np.array([s if a else 1.0 for a in amp])
        zz = f(t, *pb); r2 = 1 - ((z - zz)**2).sum()/((z - z.mean())**2).sum()
        out[name] = {"names": pn, "best": list(pb*scale),
                     "lo": [r[0]*sc for r, sc in zip(rng, scale)],
                     "hi": [r[1]*sc for r, sc in zip(rng, scale)],
                     "chi2": chi2, "r2": float(r2)}
    return out

def boxes_overlap(a, b):
    for name in a:
        if a[name] is None or b.get(name) is None: return False
        for l1, h1, l2, h2 in zip(a[name]["lo"], a[name]["hi"], b[name]["lo"], b[name]["hi"]):
            if h1 < l2 or h2 < l1: return False
    return True

def evaluate(theta, fits, mu, se, cyts, names, bounds):
    n = len(fits)
    tn = np.array([[(th[i]-bounds[p]["low"])/(bounds[p]["high"]-bounds[p]["low"]) for i, p in enumerate(names)]
                   for th in theta])
    use = [c for c, nm in enumerate(cyts) if nm in CYT_SM]
    acc, direct = [], []
    for k in range(n):
        A = [j for j in range(n) if j != k and boxes_overlap(fits[k], fits[j])]
        acc.append(A)
        w = np.array([np.nansum(((mu[j][:, use]-mu[k][:, use])/np.maximum(se[k][:, use], 1e-30))**2)
                      if j != k else np.inf for j in range(n)])
        direct.append(list(np.argsort(w)[:max(len(A), 1)]))
    def summarise(S):
        res = {}
        nonempty = [k for k in range(n) if len(S[k]) > 0]
        for i, p in enumerate(names):
            est = np.array([np.median(tn[S[k], i]) for k in nonempty])
            tru = tn[nonempty, i]
            cov = np.mean([tn[S[k], i].min() <= tn[k, i] <= tn[S[k], i].max() for k in nonempty])
            sd_ratio = np.mean([tn[S[k], i].std() for k in nonempty if len(S[k]) > 1]) / tn[:, i].std()
            r2 = 1 - ((tru-est)**2).sum()/((tru-tru.mean())**2).sum()
            res[p] = {"r2_median_accepted": float(r2), "coverage": float(cov), "sd_ratio": float(sd_ratio)}
        prod = tn[:, names.index("init_ec")] + tn[:, names.index("keil8")] if "init_ec" in names else None
        return res
    out = {"n_points": n,
           "mean_accepted": float(np.mean([len(a) for a in acc])),
           "frac_empty": float(np.mean([len(a) == 0 for a in acc])),
           "smore": summarise(acc), "direct": summarise(direct),
           "accepted_sets": acc}
    if "init_ec" in names and "keil8" in names:
        ie, k8 = names.index("init_ec"), names.index("keil8")
        P = np.log(theta[:, ie]) + np.log(theta[:, k8])            # log of the product
        ne = [k for k in range(n) if len(acc[k]) > 1]
        out["ridge"] = {"sd_ratio_log_product": float(np.mean([P[acc[k]].std() for k in ne]) / P.std()),
                        "sd_ratio_log_init_ec": float(np.mean([np.log(theta[acc[k], ie]).std() for k in ne]) / np.log(theta[:, ie]).std()),
                        "sd_ratio_log_keil8": float(np.mean([np.log(theta[acc[k], k8]).std() for k in ne]) / np.log(theta[:, k8]).std())}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates-npz", required=True)   # from collect_replicates.py
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=20000)
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--n-sd", type=float, default=2.0)   # widening of interpolated bounds by GP uncertainty
    a = ap.parse_args()
    D = np.load(a.replicates_npz, allow_pickle=True)
    Y, runs, cyts = D["Y"], [str(r) for r in D["runs"]], [str(c) for c in D["cyts"]]   # Y: (n, R, T, C)
    m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
    pmap = {r["run_id"]: r["params"] for r in m["runs"]}
    theta_all = np.array([[pmap[r][p] for p in names] for r in runs], float)
    theta = theta_all
    t = np.linspace(0, 1, Y.shape[2])
    nrep = np.sum(np.isfinite(Y[:, :, 0, 0]), axis=1)
    mu = np.nanmean(Y, axis=1)
    se = np.nanstd(Y, axis=1, ddof=1) / np.sqrt(nrep)[:, None, None]
    print(f"{len(runs)} points, replicates per point: min {nrep.min()} max {nrep.max()}", flush=True)
    cache = a.out + ".fits.json"
    if os.path.exists(cache):                       # stage 2 is the slow part: reuse it
        fits = [json.load(open(cache))[r] for r in runs]
        print(f"loaded SM fits and profile-likelihood bounds from {cache}", flush=True)
    else:
        with Pool(a.procs) as pool:
            fits = pool.map(fit_point, [(mu[i], se[i], t, cyts) for i in range(len(runs))])
        json.dump({r: f for r, f in zip(runs, fits)}, open(cache, "w"), default=float)
        print(f"saved SM fits to {cache}", flush=True)
    for name in CYT_SM:
        r2 = [f[name]["r2"] for f in fits if f.get(name)]
        print(f"  SM fit {name:5s} ({CYT_SM[name]}): median R2 {np.median(r2):.3f}, min {np.min(r2):.3f}", flush=True)
    res = evaluate(theta, fits, mu, se, cyts, names, bounds)
    sub = a.params or names                     # ABM parameters of interest (Jain: a handful at a time)
    idx = [names.index(p) for p in sub]
    C, X, acc = jain_region(theta_all[:, idx], fits, runs, sub, bounds, n_cand=a.n_cand, procs=a.procs, n_sd=a.n_sd)
    res["jain"] = summarise_region(C, X, acc, sub, theta_all[:, idx]); res["jain"]["params"] = sub
    J = res["jain"]
    print(f"[Jain 2022 region] folds with a non-empty region: {100*(1-J['frac_empty']):.0f}%")
    print(f"{'param':9s} {'R2':>7s} {'cov95':>6s} {'sd ratio':>9s}")
    for p in sub:
        q = J["per_param"][p]; print(f"{p:9s} {q['r2_median_accepted']:7.3f} {q['coverage95']:6.2f} {q['sd_ratio']:9.2f}")
    if "ridge" in J: print("ridge:", {k: round(v, 3) for k, v in J["ridge"].items()})
    res["fits"] = {r: f for r, f in zip(runs, fits)}
    res["replicates_per_point"] = nrep.tolist()
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[Bergman 2024 pointwise acceptance, for reference] accepted sweep points per fold: mean {res['mean_accepted']:.1f} of {len(runs)-1}, empty in {100*res['frac_empty']:.0f}% of folds")
    if res["frac_empty"] < 1:
        print(f"{'param':9s} {'R2':>9s} {'cover':>6s} {'sd ratio':>9s}")
        for p in names:
            q = res["smore"][p]; print(f"{p:9s} {q['r2_median_accepted']:9.3f} {q['coverage']:6.2f} {q['sd_ratio']:9.2f}")



# ---------------------------------------------------------------------------
# Jain et al. (2022), step 4 and 6: reconstruct the upper and lower 95% bound
# hypersurfaces of every SM parameter as functions of the ABM parameters, then
# infer the region of ABM parameter space whose predicted SM boxes intersect
# the data box.  In 10 dimensions with 100 sweep points the hypersurfaces are
# interpolated with Gaussian processes (Jain et al. use bilinear interpolation
# in 2D); candidate ABM parameter vectors are a dense quasi-random sample.
# ---------------------------------------------------------------------------
def _sm_bound_matrix(fits, runs):
    keys = []
    for name in CYT_SM:
        f0 = next((f[name] for f in fits if f.get(name)), None)
        if f0 is None: continue                 # cytokine absent from this data set
        for j, pn in enumerate(f0["names"]):
            keys.append((name, j, pn))
    L = np.array([[fits[i][n]["lo"][j] if fits[i].get(n) else np.nan for (n, j, _) in keys] for i in range(len(runs))])
    U = np.array([[fits[i][n]["hi"][j] if fits[i].get(n) else np.nan for (n, j, _) in keys] for i in range(len(runs))])
    return keys, L, U

def _tf(x, lo_bound):          # log-transform positive parameters for interpolation
    if lo_bound < 0: return x
    x = np.asarray(x, float); pos = x[x > 0]
    floor = 1e-3*np.median(pos) if pos.size else 1e-300   # a bound at zero = open on that side
    return np.log(np.maximum(x, floor))


_G = {}

def _run_fold(arg):
    """One leave-one-out fold of the Jain region search (uses the shared _G)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    idx, kk = arg
    X, C, Lt, Ut, kern = _G["X"], _G["C"], _G["Lt"], _G["Ut"], _G["kern"]
    n_sd, n_walk, n_anneal, n_sample = _G["n_sd"], _G["n_walk"], _G["n_anneal"], _G["n_sample"]
    rng = np.random.default_rng(_G["seed"] + 1000 + kk)
    n, nq = X.shape[0], Lt.shape[1]
    tr = np.arange(n) != kk
    models = []
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    for q in range(nq):
        pair = []
        for M in (Lt, Ut):
            y = M[tr, q]; mu_, sd_ = y.mean(), y.std() + 1e-12
            # hyperparameters are fitted on the 99 training points of this fold only
            k = ConstantKernel(1.0)*Matern(np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-2)
            pair.append((GaussianProcessRegressor(k, n_restarts_optimizer=1, random_state=0).fit(X[tr], (y-mu_)/sd_), mu_, sd_))
        models.append(pair)
    # the data box: the held-out point's own bounds, or (surrogate-in-the-loop) the
    # bounds fitted to another source's trajectory for the same point
    dl, du = (_G["Ld"][kk], _G["Ud"][kk]) if "Ld" in _G else (Lt[kk], Ut[kk])
    dw = np.maximum(du - dl, 1e-9)
    def violation(P):
        v = np.zeros(len(P))
        for q, ((gl, ml, sl), (gu, mu2, su)) in enumerate(models):
            pl, spl = gl.predict(P, return_std=True); pu, spu = gu.predict(P, return_std=True)
            pl, pu = pl*sl + ml, pu*su + mu2
            cl = np.minimum(pl, pu) - n_sd*spl*sl
            cu = np.maximum(pl, pu) + n_sd*spu*su
            v += (np.maximum(0, dl[q] - cu) + np.maximum(0, cl - du[q]))/dw[q]
        return v
    # 1) screen a quasi-random sample, 2) anneal walkers into the region where every
    #    interval overlaps (violation = 0), 3) random walk uniformly inside that region
    v0 = violation(C); order = np.argsort(v0)[:n_walk]
    W = C[order].copy(); vW = v0[order].copy()
    for T in np.geomspace(1.0, 1e-4, n_anneal):
        P = np.clip(W + 0.05*rng.standard_normal(W.shape), 0, 1); vP = violation(P)
        with np.errstate(over="ignore"):
            acc = (vP <= vW) | (rng.random(len(W)) < np.exp(-np.maximum(vP - vW, 0)/T))
        W[acc], vW[acc] = P[acc], vP[acc]
    W = W[vW == 0]
    samples = []
    if len(W):
        for _ in range(n_sample):
            P = np.clip(W + 0.02*rng.standard_normal(W.shape), 0, 1); ok = violation(P) == 0
            W[ok] = P[ok]; samples.append(W.copy())
    return idx, (np.concatenate(samples) if samples else np.empty((0, X.shape[1])))

def jain_region(theta, fits, runs, names, bounds, n_cand=20000, seed=0, n_sd=2.0, n_walk=200, n_anneal=150, n_sample=100, folds=None, procs=8, data_fits=None):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from scipy.stats import qmc
    lo = np.array([bounds[p]["low"] for p in names]); hi = np.array([bounds[p]["high"] for p in names])
    X = (theta - lo)/(hi - lo)
    C = qmc.Sobol(len(names), scramble=True, seed=seed).random(n_cand)
    keys, L, U = _sm_bound_matrix(fits, runs)
    lob = {(n, j): SM[CYT_SM[n]][4][j] for (n, j, _) in keys}
    Lt = np.column_stack([_tf(L[:, q], lob[(n, j)]) for q, (n, j, _) in enumerate(keys)])
    Ut = np.column_stack([_tf(U[:, q], lob[(n, j)]) for q, (n, j, _) in enumerate(keys)])
    kern = None                                   # surfaces are fitted inside every fold (no leakage)
    n = len(runs)
    _G.clear()
    _G.update(X=X, C=C, Lt=Lt, Ut=Ut, kern=kern, n_sd=n_sd, n_walk=n_walk,
              n_anneal=n_anneal, n_sample=n_sample, seed=seed)
    if data_fits is not None:                   # data boxes from another source (e.g. the surrogate)
        _, Ld, Ud = _sm_bound_matrix(data_fits, runs)
        _G["Ld"] = np.column_stack([_tf(Ld[:, q], lob[(nn, j)]) for q, (nn, j, _) in enumerate(keys)])
        _G["Ud"] = np.column_stack([_tf(Ud[:, q], lob[(nn, j)]) for q, (nn, j, _) in enumerate(keys)])
    fl = list(range(n) if folds is None else folds)
    accepted = [None]*len(fl)
    # leave-one-out folds are independent: run them in parallel (fork inherits _G)
    with Pool(procs) as pool:
        for done, (idx, A) in enumerate(pool.imap_unordered(_run_fold, list(enumerate(fl))), 1):
            accepted[idx] = A
            if done % 10 == 0 or done == len(fl):
                print(f"  region: {done}/{len(fl)} folds done", flush=True)
    return C, X, accepted

def summarise_region(C, X, accepted, names, theta):
    n = len(accepted); res = {}
    ne = [k for k in range(n) if len(accepted[k]) > 1]
    for i, p in enumerate(names):
        est = np.array([np.median(accepted[k][:, i]) for k in ne]); tru = X[ne, i]
        r2 = 1 - ((tru-est)**2).sum()/((tru-tru.mean())**2).sum() if ne else float("nan")
        cov = np.mean([np.quantile(accepted[k][:, i], .025) <= X[k, i] <= np.quantile(accepted[k][:, i], .975) for k in ne]) if ne else float("nan")
        sdr = np.mean([accepted[k][:, i].std() for k in ne]) / np.sqrt(1/12) if ne else float("nan")
        res[p] = {"r2_median_accepted": float(r2), "coverage95": float(cov), "sd_ratio": float(sdr)}
    out = {"per_param": res, "mean_accepted_frac": float(len(ne)/n),
           "frac_empty": float(np.mean([len(a) <= 1 for a in accepted]))}
    if "init_ec" in names and "keil8" in names:
        ie, k8 = names.index("init_ec"), names.index("keil8")
        out["ridge"] = {"sd_ratio_sum": float(np.mean([(accepted[k][:, ie] + accepted[k][:, k8]).std() for k in ne]) / np.sqrt(2/12)) if ne else float("nan"),
                        "sd_ratio_difference": float(np.mean([(accepted[k][:, ie] - accepted[k][:, k8]).std() for k in ne]) / np.sqrt(2/12)) if ne else float("nan")}
    return out

if __name__ == "__main__":
    main()
