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
         --sigma-mode controls what happens when the replicate standard error
         is far below the SM approximation error (volume-averaged IL-8 is
         deterministic to ~1e-5 relative):
           paper       replicate SE, except that a cytokine whose median SE is
                       below DET_TOL of the trajectory maximum uses the RMS
                       residual of an unweighted fit instead (the choice stated
                       in the manuscript; the v1 code tested SE <= 1e-12 and
                       therefore never took this branch)
           quadrature  sqrt(SE^2 + RMS_fit^2) for every cytokine: replicate
                       noise and SM approximation error added in quadrature
           se          replicate SE only (v1 behaviour, Jain eq. 2 literally)
Stage 3  leave-one-out: the held-out point's own SM bounds play the role of
         the data-derived set Phi (Bergman 2024).  Jain step 4/6: the lower and
         upper bound of every SM parameter is interpolated over ABM parameter
         space with a Gaussian process, and the admissible region is the set of
         ABM parameter vectors whose predicted (GP-widened) box intersects the
         data box in every SM parameter.
         --sampler controls how that region is sampled:
           qmc         (default) evaluate the overlap condition on a scrambled
                       Sobol sequence, drawn in batches until at least --n-min
                       points are accepted or --n-cand-max points have been
                       screened.  Accepted points are a uniform (quasi-random)
                       sample of the region by construction, so widths and
                       coverage are unbiased; the acceptance fraction is the
                       region's volume fraction of the sweep box.  If fewer
                       than --n-min points are found, an exact hit-and-run
                       Markov chain (uniform target, no clipping) continues
                       from them.
           walk        v1 procedure (simulated annealing + random walk with
                       proposals clipped to the box).  Kept for comparison
                       only: clipping puts point mass on the box boundary and
                       biases the width estimate upwards.
"""
import numpy as np, json, sys, argparse, os, time
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
DET_TOL = 1e-4          # median replicate SE / trajectory maximum below which a cytokine counts as deterministic
SIGMA_MODE = "paper"    # module default; main() and the callers set it from the command line
SE_FLOOR_FRAC = 0.05    # standard errors below this fraction of their median are raised to it (Jain-style fit stabiliser)

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
    then bisect for the crossing point.  The other parameters are re-optimised
    at every value of parameter j (Raue et al. 2009)."""
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
    """One sweep point: returns per cytokine best params and 95% bounds (physical units).

    args = (mu, se, t, cyts[, sigma_mode[, extra_rel_sigma]])
      mu, se           (T, C) replicate mean and its standard error
      sigma_mode       'paper' | 'quadrature' | 'se'  (default: module SIGMA_MODE)
      extra_rel_sigma  optional (C,) or (T, C) array of additional errors relative to the
                       trajectory maximum, added in quadrature to sigma; used in the
                       surrogate-in-the-loop test to widen the data box by the neural
                       surrogate's own error
    """
    mu, se, t, cyts = args[:4]
    mode = args[4] if len(args) > 4 and args[4] is not None else SIGMA_MODE
    extra = args[5] if len(args) > 5 else None
    out = {}
    for c, name in enumerate(cyts):
        if name not in CYT_SM: continue
        f, pn, amp, p0s, lo, hi = SM[CYT_SM[name]]
        y = mu[:, c]; s = max(np.abs(y).max(), 1e-30)
        z = y/s
        sig_se = se[:, c]/s
        pos = sig_se[sig_se > 0]
        floor = max(SE_FLOOR_FRAC*np.median(pos), 1e-6) if pos.size else 1e-6
        sig_se = np.maximum(sig_se, floor)
        deterministic = bool(np.median(se[:, c]/s) < DET_TOL)
        # RMS residual of an unweighted fit = SM approximation error on this trajectory
        _, pu = wfit(f, t, z, np.ones_like(z), p0s, lo, hi)
        rms = float(np.sqrt(np.mean((z - f(t, *pu))**2))) if pu is not None else 1e-3
        rms = max(rms, 1e-6)
        if mode == "se":
            sig = sig_se
        elif mode == "paper":
            sig = np.full_like(z, rms) if deterministic else sig_se
        elif mode == "quadrature":
            sig = np.sqrt(sig_se**2 + rms**2)
        else:
            raise ValueError(f"unknown sigma mode {mode}")
        if extra is not None:                  # (C,) constant or (T, C) per-frame relative tolerance
            e = np.asarray(extra, float); e = e[:, c] if e.ndim == 2 else np.full_like(z, e[c])
            sig = np.sqrt(sig**2 + e**2)
        chi2, pb = wfit(f, t, z, sig, p0s, lo, hi)
        if pb is None:
            out[name] = None; continue
        rng = [profile_bounds(f, t, z, sig, pb, chi2, j, lo, hi, p0s) for j in range(len(pn))]
        # identifiability of the SM fit itself: a best-fit value at an upper limit of the allowed range, or at a
        # lower limit that is not a natural boundary (zero), or a profile interval that spans the whole allowed
        # range, means the trajectory does not determine that SM parameter (e.g. a plateau that is never reached)
        tol = lambda lim: 1e-6*max(1.0, abs(lim))
        unid = [bool(pb[j] >= hi[j] - tol(hi[j]) or (lo[j] != 0 and pb[j] <= lo[j] + tol(lo[j]))
                     or (rng[j][0] <= lo[j] + tol(lo[j]) and rng[j][1] >= hi[j] - tol(hi[j]))) for j in range(len(pn))]
        scale = np.array([s if a else 1.0 for a in amp])
        zz = f(t, *pb); r2 = 1 - ((z - zz)**2).sum()/((z - z.mean())**2).sum()
        out[name] = {"names": pn, "best": list(pb*scale),
                     "lo": [r[0]*sc for r, sc in zip(rng, scale)],
                     "hi": [r[1]*sc for r, sc in zip(rng, scale)],
                     "chi2": chi2, "r2": float(r2), "rms_fit": rms, "deterministic": deterministic,
                     "unidentified": unid, "identified": not any(unid),
                     "sigma_mode": mode, "median_rel_se": float(np.median(se[:, c]/s)),
                     "median_rel_sigma": float(np.median(sig))}
    return out

def boxes_overlap(a, b):
    for name in a:
        if a[name] is None or b.get(name) is None: return False
        for l1, h1, l2, h2 in zip(a[name]["lo"], a[name]["hi"], b[name]["lo"], b[name]["hi"]):
            if h1 < l2 or h2 < l1: return False
    return True

def evaluate(theta, fits, mu, se, cyts, names, bounds):
    """Bergman et al. (2024) pointwise acceptance of simulated sweep points (no
    interpolation) and the direct trajectory-distance method, for reference."""
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
            if not nonempty:
                res[p] = {"r2_median_accepted": float("nan"), "coverage": float("nan"), "sd_ratio": float("nan")}; continue
            est = np.array([np.median(tn[S[k], i]) for k in nonempty])
            tru = tn[nonempty, i]
            cov = np.mean([tn[S[k], i].min() <= tn[k, i] <= tn[S[k], i].max() for k in nonempty])
            multi = [k for k in nonempty if len(S[k]) > 1]
            sd_ratio = np.mean([tn[S[k], i].std() for k in multi]) / tn[:, i].std() if multi else float("nan")
            r2 = 1 - ((tru-est)**2).sum()/((tru-tru.mean())**2).sum() if len(tru) > 1 else float("nan")
            res[p] = {"r2_median_accepted": float(r2), "coverage": float(cov), "sd_ratio": float(sd_ratio)}
        return res
    out = {"n_points": n,
           "mean_accepted": float(np.mean([len(a) for a in acc])),
           "frac_empty": float(np.mean([len(a) == 0 for a in acc])),
           "smore": summarise(acc), "direct": summarise(direct),
           "accepted_sets": acc}
    return out

# ---------------------------------------------------------------------------
# Jain et al. (2022), step 4 and 6: reconstruct the upper and lower 95% bound
# hypersurfaces of every SM parameter as functions of the ABM parameters, then
# infer the region of ABM parameter space whose predicted SM boxes intersect
# the data box.  In 10 dimensions with 100 sweep points the hypersurfaces are
# interpolated with Gaussian processes (Jain et al. use bilinear interpolation
# in 2D); the region is sampled by rejection on a quasi-random sequence.
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

def _sm_best_matrix(fits, runs, keys):
    """Best-fit SM parameters (transformed like the bounds) as features for the inverse-regression baseline."""
    B = np.array([[fits[i][n]["best"][j] if fits[i].get(n) else np.nan for (n, j, _) in keys] for i in range(len(runs))])
    lob = {(n, j): SM[CYT_SM[n]][4][j] for (n, j, _) in keys}
    F = np.column_stack([_tf(B[:, q], lob[(n, j)]) for q, (n, j, _) in enumerate(keys)])
    col_med = np.nanmedian(F, axis=0)
    return np.where(np.isfinite(F), F, col_med)

MIN_LIBRARY = 10   # a bound surface needs at least this many identified library points, otherwise it is dropped

def _fit_bound_surfaces(X, Lt, Ut, tr, unid=None):
    """One GP per (SM parameter, lower/upper bound), fitted on the training rows `tr`.
    The predictive standard deviation returned by sklearn includes the fitted
    WhiteKernel noise term, so the n_sd widening covers interpolation error and
    the scatter of the profile-likelihood bounds about the surface."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    models = []
    for q in range(Lt.shape[1]):
        # library points at which this SM parameter is not identified (see fit_point) carry no information
        # about its bound surfaces and would enter the GP as outliers; they are left out of this surface only
        rows = tr & np.isfinite(Lt[:, q]) & np.isfinite(Ut[:, q])
        if unid is not None: rows &= ~unid[:, q]
        if rows.sum() < MIN_LIBRARY:
            models.append(None); continue
        pair = []
        for M in (Lt, Ut):
            y = M[rows, q]; mu_, sd_ = y.mean(), y.std() + 1e-12
            k = ConstantKernel(1.0)*Matern(np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-2)
            pair.append((GaussianProcessRegressor(k, n_restarts_optimizer=1, random_state=0).fit(X[rows], (y-mu_)/sd_), mu_, sd_))
        models.append(pair)
    return models

def _make_violation(models, dl, du, n_sd):
    dw = np.maximum(du - dl, 1e-9)
    def violation(P):
        """Sum over SM parameters of the gap between the predicted (widened) interval
        and the data interval, normalised by the data interval; 0 iff every interval
        overlaps (Jain step 6)."""
        v = np.zeros(len(P))
        for q, mq in enumerate(models):
            if mq is None or not (np.isfinite(dl[q]) and np.isfinite(du[q])): continue   # no surface / no data interval: no constraint
            (gl, ml, sl), (gu, mu2, su) = mq
            pl, spl = gl.predict(P, return_std=True); pu, spu = gu.predict(P, return_std=True)
            pl, pu = pl*sl + ml, pu*su + mu2
            cl = np.minimum(pl, pu) - n_sd*spl*sl
            cu = np.maximum(pl, pu) + n_sd*spu*su
            v += (np.maximum(0, dl[q] - cu) + np.maximum(0, cl - du[q]))/dw[q]
        return v
    return violation

def _hit_and_run(W, violation, n_steps, rng, n_grid=48, n_bisect=5, cov=None):
    """Hit-and-run sampler for the uniform distribution on the region {violation == 0}
    inside the unit box, with a 'slice' step.  For every walker: draw a random direction,
    take the chord of the box along it, locate the feasible part of the chord on a grid of
    n_grid points (jittered, including both chord ends and the current point) refined by
    n_bisect bisections at the ends of every feasible run, and draw the next point
    uniformly on that feasible set.  The feasible set of a line is the same from every
    point on the line, so the kernel is symmetric and the uniform distribution on the
    region is invariant; nothing is clipped and no proposal leaves the box.  A final
    membership check rejects the rare draw outside the region.
    Directions: if `cov` is given, half of the walkers per step draw their direction from
    N(0, cov) (the shape of the region estimated from the starting points) and half
    isotropically.  The direction distribution is fixed and symmetric (u and -u equally
    likely), so the uniform distribution stays invariant; aligned directions only make
    the chain mix faster in thin, oblique regions.
    Returns samples of shape (n_steps, walkers, d) and the move rate."""
    W = W.copy(); m, d = W.shape; samples = []; n_acc = 0
    for _ in range(n_steps):
        u = rng.standard_normal((m, d))
        if cov is not None:
            al = rng.random(m) < 0.5
            u[al] = rng.multivariate_normal(np.zeros(d), cov, size=int(al.sum()), method="eigh")
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (0 - W)/u; t2 = (1 - W)/u
        tneg = np.where(u != 0, np.minimum(t1, t2), -np.inf); tpos = np.where(u != 0, np.maximum(t1, t2), np.inf)
        t_lo = tneg.max(axis=1); t_hi = tpos.min(axis=1)
        g = np.linspace(0, 1, n_grid)[None, :].repeat(m, 0)
        g[:, 1:-1] += (rng.random((m, n_grid - 2)) - 0.5)/n_grid
        tg = t_lo[:, None] + g*(t_hi - t_lo)[:, None]
        tg = np.sort(np.concatenate([tg, np.zeros((m, 1))], axis=1), axis=1)
        P = W[:, None, :] + tg[..., None]*u[:, None, :]
        feas = (violation(P.reshape(-1, d)) == 0).reshape(m, -1)
        feas[np.arange(m), np.argmin(np.abs(tg), axis=1)] = True
        runs, edges = [[] for _ in range(m)], []
        for i in range(m):
            f = feas[i]; j = 0; ng = len(f)
            while j < ng:
                if not f[j]: j += 1; continue
                k = j
                while k + 1 < ng and f[k + 1]: k += 1
                a, b = tg[i, j], tg[i, k]
                ea = len(edges) if j > 0 else None
                if j > 0: edges.append([i, tg[i, j], tg[i, j - 1]])
                eb = len(edges) if k + 1 < ng else None
                if k + 1 < ng: edges.append([i, tg[i, k], tg[i, k + 1]])
                runs[i].append([a, b, ea, eb]); j = k + 1
        E = np.array(edges, float) if edges else np.empty((0, 3))
        for _b in range(n_bisect):
            if not len(E): break
            mid = 0.5*(E[:, 1] + E[:, 2]); wi = E[:, 0].astype(int)
            ok = violation(W[wi] + mid[:, None]*u[wi]) == 0
            E[ok, 1] = mid[ok]; E[~ok, 2] = mid[~ok]
        newW = W.copy(); moved = np.zeros(m, bool)
        for i in range(m):
            iv = []
            for a, b, ea, eb in runs[i]:
                if ea is not None: a = E[ea, 1]
                if eb is not None: b = E[eb, 1]
                iv.append((a, max(b, a)))
            L = np.array([b - a for a, b in iv]); tot = L.sum()
            if tot <= 0: continue
            r = rng.random()*tot; c = 0.0; tt = 0.0
            for (a, b), l in zip(iv, L):
                if r <= c + l: tt = a + (r - c); break
                c += l
            newW[i] = W[i] + tt*u[i]; moved[i] = True
        chk = moved & (violation(newW) == 0) & np.all((newW >= 0) & (newW <= 1), axis=1)
        W[chk] = newW[chk]; n_acc += chk.sum(); samples.append(W.copy())
    return np.stack(samples), n_acc/(n_steps*m)

def _rhat(S):
    """Split R-hat (Gelman-Rubin) per coordinate over walkers; S has shape (steps, walkers, d)."""
    n = (S.shape[0]//2)*2
    if n < 4: return float("nan")
    C = np.concatenate([S[:n//2], S[n//2:n]], axis=1)
    nn = C.shape[0]; cm = C.mean(0); cv = C.var(0, ddof=1)
    B = nn*cm.var(0, ddof=1); Wv = cv.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(((nn-1)/nn*Wv + B/nn)/Wv)
    r = r[np.isfinite(r)]
    return float(r.max()) if r.size else float("nan")

def _anneal(W, vW, violation, rng, n_anneal=150, step=0.05):
    """Simulated annealing towards violation 0, proposals outside the box rejected
    (no clipping).  Only used to find a starting point when the region is too small
    for the quasi-random screen; samples are then drawn by hit-and-run."""
    W, vW = W.copy(), vW.copy()
    for T in np.geomspace(1.0, 1e-4, n_anneal):
        P = W + step*rng.standard_normal(W.shape)
        inb = np.all((P >= 0) & (P <= 1), axis=1)
        vP = np.full(len(P), np.inf)
        if inb.any(): vP[inb] = violation(P[inb])
        with np.errstate(over="ignore", invalid="ignore"):
            ok = inb & ((vP <= vW) | (rng.random(len(W)) < np.exp(-np.maximum(vP - vW, 0)/T)))
        W[ok], vW[ok] = P[ok], vP[ok]
    return W[vW == 0]

def _grid_moves(W, violation, levels, rng, weight):
    """One random-scan coordinate update per walker on a product grid: every walker picks a
    coordinate, all levels of that coordinate are evaluated with the others fixed, and the
    new level is drawn with probability proportional to weight(violation)."""
    m, d = W.shape; j = rng.integers(d, size=m)
    C = np.concatenate([np.repeat(W[i:i+1], len(levels[j[i]]), 0) for i in range(m)])
    own = np.concatenate([np.full(len(levels[j[i]]), i) for i in range(m)]); pos = 0
    for i in range(m):
        C[pos:pos+len(levels[j[i]]), j[i]] = levels[j[i]]; pos += len(levels[j[i]])
    v = violation(C); W = W.copy()
    for i in range(m):
        sel = np.where(own == i)[0]; w = weight(v[sel])
        if w.sum() > 0: W[i] = C[sel[rng.choice(len(sel), p=w/w.sum())]]
    return W

def _grid_gibbs(W, violation, levels, n_steps, rng):
    """Random-scan Gibbs sampler for the uniform distribution on the admissible points of a
    product grid: the new level of the chosen coordinate is uniform over its admissible
    levels (the current level is always one of them).  Returns (steps, walkers, d)."""
    out = []
    for _ in range(n_steps):
        W = _grid_moves(W, violation, levels, rng, lambda v: (v == 0).astype(float)); out.append(W.copy())
    return np.stack(out)

def _grid_anneal(W, violation, levels, rng, n_anneal=150):
    """Annealing on the grid towards violation 0 (only to find starting points)."""
    for T in np.geomspace(1.0, 1e-4, n_anneal):
        W = _grid_moves(W, violation, levels, rng, lambda v, T=T: np.exp(-(v - v.min())/T))
    return W[violation(W) == 0]

def _run_fold(arg):
    """One leave-one-out fold (uses the shared _G): bound surfaces fitted without the
    held-out point, diagnostics at the held-out point, admissible-region sample."""
    idx, kk = arg
    X, Lt, Ut = _G["X"], _G["Lt"], _G["Ut"]
    n_sd, sampler, seed = _G["n_sd"], _G["sampler"], _G["seed"]
    rng = np.random.default_rng(seed + 1000 + kk)
    n, d = X.shape
    tr = np.arange(n) != kk
    models = _fit_bound_surfaces(X, Lt, Ut, tr, _G.get("unid"))   # hyperparameters fitted on the training points of this fold only
    # data box: the held-out point's own bounds, or (surrogate-in-the-loop) bounds fitted to another source
    dl, du = (_G["Ld"][kk], _G["Ud"][kk]) if "Ld" in _G else (Lt[kk], Ut[kk])
    violation = _make_violation(models, dl, du, n_sd)
    info = {"fold": int(kk), "sampler": sampler}
    # (a) leave-one-out check of the bound surfaces against the held-out library value
    xk = X[kk:kk+1]; diag = {"lo_pred": [], "lo_sd": [], "hi_pred": [], "hi_sd": []}
    unid_k = _G["unid"][kk] if "unid" in _G else np.zeros(Lt.shape[1], bool)
    for q, mq in enumerate(models):
        if mq is None or unid_k[q]:                                    # surface dropped, or held-out value undefined
            for key in ("lo_pred", "lo_sd", "hi_pred", "hi_sd"): diag[key].append(float("nan"))
            continue
        (gl, ml, sl), (gu, mu2, su) = mq
        a, sa = gl.predict(xk, return_std=True); b, sb = gu.predict(xk, return_std=True)
        diag["lo_pred"].append(float(a[0]*sl + ml)); diag["lo_sd"].append(float(sa[0]*sl))
        diag["hi_pred"].append(float(b[0]*su + mu2)); diag["hi_sd"].append(float(sb[0]*su))
    diag["lo_true"] = Lt[kk].tolist(); diag["hi_true"] = Ut[kk].tolist()
    info["surfaces_dropped"] = [int(q) for q, mq in enumerate(models) if mq is None]
    info["surface_loo"] = diag
    # (b) is the true parameter vector itself admissible (joint containment)?
    info["truth_admissible"] = bool(violation(xk)[0] == 0)
    # (c) model-free reference: GP regression from the SM best-fit parameters to each ABM parameter
    if "F_lib" in _G:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        Fl = _G["F_lib"]; Fd = _G.get("F_data", Fl)
        mf, sf = Fl[tr].mean(0), Fl[tr].std(0) + 1e-12
        Z, zk = (Fl[tr] - mf)/sf, (Fd[kk:kk+1] - mf)/sf
        pred = []
        for i in range(d):
            y = X[tr, i]; m_, s_ = y.mean(), y.std() + 1e-12
            kern = ConstantKernel(1.0)*Matern(np.ones(Z.shape[1]), nu=2.5) + WhiteKernel(1e-1)
            g = GaussianProcessRegressor(kern, n_restarts_optimizer=0, random_state=0).fit(Z, (y - m_)/s_)
            pred.append(float(np.clip(g.predict(zk)[0]*s_ + m_, 0.0, 1.0)))   # prediction restricted to the sweep box
        info["inverse_pred"] = pred
    # (d) sample the admissible region
    if sampler == "design":                                    # finite design: evaluate every realisable input vector exactly
        C = _G["cands"]; v = violation(C); A = C[v == 0]
        info.update(n_screened=int(len(C)), n_accepted_qmc=int(len(A)), volume_fraction=float(len(A)/len(C)),
                    found_by="design" if len(A) else None, mcmc_steps=0, rhat_max=float("nan"))
    elif sampler == "grid":                                      # discrete design: the inputs can only take observed levels
        lv = _G["levels"]; n_min, n_batch, n_max, n_keep = _G["n_min"], _G["n_cand"], _G["n_cand_max"], _G["n_keep"]
        n_grid_total = float(np.prod([len(l) for l in lv]))
        acc, n_screened, best_P, best_v = [], 0, np.empty((0, d)), np.empty(0)
        while True:
            C = np.column_stack([rng.choice(l, n_batch) for l in lv]); n_screened += n_batch
            v = violation(C); acc.append(C[v == 0])
            o = np.argsort(np.concatenate([best_v, v]))[:200]
            best_P = np.concatenate([best_P, C])[o]; best_v = np.concatenate([best_v, v])[o]
            if sum(len(a) for a in acc) >= n_min or n_screened >= n_max or n_screened >= 4*n_grid_total: break
        A = np.concatenate(acc); n_qmc = len(A)
        info.update(n_screened=int(n_screened), n_accepted_qmc=int(n_qmc), volume_fraction=float(n_qmc/n_screened),
                    grid_points=n_grid_total, found_by="grid" if n_qmc else None, mcmc_steps=0, rhat_max=float("nan"))
        seeds, burn = A, 0
        if n_qmc == 0:
            seeds = _grid_anneal(best_P, violation, lv, rng)
            if len(seeds): info["found_by"] = "anneal"; burn = 1
        if 0 < len(seeds) and len(A) < n_min:
            nw = _G["n_walkers"]
            W0 = seeds[rng.choice(len(seeds), nw, replace=len(seeds) < nw)]
            n_steps = int(min(_G["mcmc_max_steps"], max(50, np.ceil(_G["mcmc_total"]/len(W0)))))
            Sx = _grid_gibbs(W0, violation, lv, n_steps, rng)
            if burn: Sx = Sx[n_steps//2:]
            info.update(mcmc_steps=n_steps, rhat_max=_rhat(Sx), n_walkers=int(len(W0)))
            A = np.concatenate([A, Sx.reshape(-1, d)])
        if len(A) > n_keep: A = A[rng.choice(len(A), n_keep, replace=False)]
    elif sampler == "qmc":
        from scipy.stats import qmc
        n_min, n_batch, n_max, n_keep = _G["n_min"], _G["n_cand"], _G["n_cand_max"], _G["n_keep"]
        eng = qmc.Sobol(d, scramble=True, seed=seed)
        acc, n_screened, best_P, best_v = [], 0, np.empty((0, d)), np.empty(0)
        while True:
            C = eng.random(n_batch); n_screened += n_batch
            v = violation(C); acc.append(C[v == 0])
            o = np.argsort(np.concatenate([best_v, v]))[:200]
            best_P = np.concatenate([best_P, C])[o]; best_v = np.concatenate([best_v, v])[o]
            if sum(len(a) for a in acc) >= n_min or n_screened >= n_max: break
        A = np.concatenate(acc); n_qmc = len(A)
        info.update(n_screened=int(n_screened), n_accepted_qmc=int(n_qmc), volume_fraction=float(n_qmc/n_screened),
                    found_by="qmc" if n_qmc else None, mcmc_steps=0, mcmc_rate=float("nan"), rhat_max=float("nan"))
        seeds, burn = A, 0
        if n_qmc == 0:                                         # region smaller than 1/n_screened of the box
            seeds = _anneal(best_P, best_v, violation, rng)
            if len(seeds):
                info["found_by"] = "anneal"; burn = 1
        if 0 < len(seeds) and len(A) < n_min:                  # too few uniform points: exact hit-and-run
            nw = _G["n_walkers"]                                # between n_walkers and 10*n_walkers chains
            W0 = seeds[rng.choice(len(seeds), nw, replace=True)] if len(seeds) < nw else \
                 seeds[rng.choice(len(seeds), min(len(seeds), 10*nw), replace=False)]
            n_steps = int(min(_G["mcmc_max_steps"], max(50, np.ceil(_G["mcmc_total"]/len(W0)))))
            cov = np.cov(seeds.T) + 1e-6*np.eye(d) if len(seeds) > d else None
            # Run the chains in rounds and extend them until the split R-hat of the second half of the chain is
            # at or below the target, or the step cap is reached.  The sample is always the second half of the
            # whole chain, so the starting points (annealed or not) never enter it.
            chunks, rates, W, steps, target = [], [], W0, 0, _G["rhat_target"]
            while True:
                Sx, rate = _hit_and_run(W, violation, n_steps, rng, cov=cov)
                chunks.append(Sx); rates.append(rate); steps += n_steps; W = Sx[-1]
                S_all = np.concatenate(chunks); S_half = S_all[len(S_all)//2:]
                rh = _rhat(S_half)
                if rh <= target or steps >= _G["mcmc_max_steps"]: break
                n_steps = int(min(2*n_steps, _G["mcmc_max_steps"] - steps))
            info.update(mcmc_steps=int(steps), mcmc_rate=float(np.mean(rates)), rhat_max=float(rh), n_walkers=int(len(W0)),
                        converged=bool(rh <= target))
            A = np.concatenate([A, S_half.reshape(-1, d)])
        if len(A) > n_keep: A = A[rng.choice(len(A), n_keep, replace=False)]
    elif sampler == "walk":                                    # v1 procedure (clipped proposals), for comparison only
        from scipy.stats import qmc
        C = qmc.Sobol(d, scramble=True, seed=seed).random(_G["n_cand"])
        n_walk, n_anneal, n_sample = _G["n_walk"], _G["n_anneal"], _G["n_sample"]
        v0 = violation(C); order = np.argsort(v0)[:n_walk]
        W = C[order].copy(); vW = v0[order].copy()
        for T in np.geomspace(1.0, 1e-4, n_anneal):
            P = np.clip(W + 0.05*rng.standard_normal(W.shape), 0, 1); vP = violation(P)
            with np.errstate(over="ignore"):
                ok = (vP <= vW) | (rng.random(len(W)) < np.exp(-np.maximum(vP - vW, 0)/T))
            W[ok], vW[ok] = P[ok], vP[ok]
        W = W[vW == 0]; samples = []
        if len(W):
            for _ in range(n_sample):
                P = np.clip(W + 0.02*rng.standard_normal(W.shape), 0, 1); ok = violation(P) == 0
                W[ok] = P[ok]; samples.append(W.copy())
        A = np.concatenate(samples) if samples else np.empty((0, d))
        info.update(n_screened=int(len(C)), n_accepted_qmc=int(np.sum(v0 == 0)), volume_fraction=float(np.mean(v0 == 0)),
                    found_by="walk" if len(A) else None)
    else:
        raise ValueError(sampler)
    return idx, A, info

def jain_region(theta, fits, runs, names, bounds, n_cand=131072, seed=0, n_sd=2.0, n_walk=200, n_anneal=150, n_sample=100,
                folds=None, procs=8, data_fits=None, sampler="qmc", n_min=300, n_cand_max=4194304, n_keep=20000,
                mcmc_total=40000, n_walkers=20, mcmc_max_steps=16000, rhat_target=1.1, inverse=True, levels=None, candidates=None):
    """Leave-one-out admissible regions.  Returns (X, accepted, info): X the sweep
    points in [0,1]^d, accepted[k] the sample of the region for fold k, info[k]
    the per-fold diagnostics.  With `levels` (one array of admissible values per
    parameter) the region is searched on the product grid of those values instead of
    the continuous box.  With `candidates` (an array of realisable input vectors, e.g. the
    conditions of a factorial experiment) the region is the subset of those vectors that is
    admissible, evaluated exactly."""
    lo = np.array([bounds[p]["low"] for p in names]); hi = np.array([bounds[p]["high"] for p in names])
    X = (theta - lo)/(hi - lo)
    keys, L, U = _sm_bound_matrix(fits, runs)
    lob = {(n, j): SM[CYT_SM[n]][4][j] for (n, j, _) in keys}
    Lt = np.column_stack([_tf(L[:, q], lob[(n, j)]) for q, (n, j, _) in enumerate(keys)])
    Ut = np.column_stack([_tf(U[:, q], lob[(n, j)]) for q, (n, j, _) in enumerate(keys)])
    n = len(runs)
    _G.clear()
    _G.update(X=X, Lt=Lt, Ut=Ut, keys=keys, n_sd=n_sd, seed=seed, sampler=sampler, n_cand=n_cand, n_min=n_min,
              n_cand_max=n_cand_max, n_keep=n_keep, mcmc_total=mcmc_total, n_walkers=n_walkers, mcmc_max_steps=mcmc_max_steps, rhat_target=rhat_target,
              n_walk=n_walk, n_anneal=n_anneal, n_sample=n_sample)
    if candidates is not None:                  # finite design: the region is a subset of the realisable input vectors
        _G["cands"] = (np.asarray(candidates, float) - lo)/(hi - lo)
        _G["sampler"] = "design"
    elif levels is not None:                    # discrete design: candidate set = product of the observed levels
        _G["levels"] = [np.unique((np.asarray(l, float) - lo[i])/(hi[i] - lo[i])) for i, l in enumerate(levels)]
        _G["sampler"] = "grid"
    _G["unid"] = np.array([[(not fits[i].get(nm)) or bool(fits[i][nm].get("unidentified", [False]*99)[j]) for (nm, j, _) in keys]
                           for i in range(len(runs))], bool)
    if inverse:
        _G["F_lib"] = _sm_best_matrix(fits, runs, keys)
    if data_fits is not None:                   # data boxes from another source (e.g. the surrogate)
        _, Ld, Ud = _sm_bound_matrix(data_fits, runs)
        _G["Ld"] = np.column_stack([_tf(Ld[:, q], lob[(nn, j)]) for q, (nn, j, _) in enumerate(keys)])
        _G["Ud"] = np.column_stack([_tf(Ud[:, q], lob[(nn, j)]) for q, (nn, j, _) in enumerate(keys)])
        if inverse: _G["F_data"] = _sm_best_matrix(data_fits, runs, keys)
    fl = list(range(n) if folds is None else folds)
    accepted, info = [None]*len(fl), [None]*len(fl)
    t0 = time.time()
    if procs > 1:
        with Pool(procs) as pool:          # folds are independent; fork inherits _G
            for done, (idx, A, inf) in enumerate(pool.imap_unordered(_run_fold, list(enumerate(fl))), 1):
                accepted[idx], info[idx] = A, inf
                if done % 10 == 0 or done == len(fl):
                    print(f"  region: {done}/{len(fl)} folds done ({time.time()-t0:.0f} s)", flush=True)
    else:
        for done, arg in enumerate(enumerate(fl), 1):
            idx, A, inf = _run_fold(arg); accepted[idx], info[idx] = A, inf
            if done % 10 == 0 or done == len(fl):
                print(f"  region: {done}/{len(fl)} folds done ({time.time()-t0:.0f} s)", flush=True)
    for inf in info:
        inf["keys"] = [f"{a}_{c}" for a, _, c in keys]
        inf["library_unidentified"] = _G["unid"].sum(0).tolist()
    return X[fl], accepted, info

def _r2(tru, est):
    tru, est = np.asarray(tru, float), np.asarray(est, float)
    return float(1 - ((tru - est)**2).sum()/((tru - tru.mean())**2).sum()) if len(tru) > 1 else float("nan")

def summarise_region(X, accepted, names, info=None, n_sd=2.0, ref_sd=None, rhat_target=1.1):
    """X: true parameters of the folds in [0,1]^d (as returned by jain_region).
    Per parameter: R2 of the region median against the truth, marginal coverage of the
    central 95% of the region (over non-empty folds, and over all folds with an empty
    region counted as a miss), width (SD of the region / SD of a uniform over the
    range, or over the grid levels if ref_sd is given), and the R2 of the model-free
    inverse regression.  Joint: share of folds in
    which the true vector itself is admissible.  Surfaces: leave-one-out R2 of every
    interpolated bound and the share of held-out bounds within n_sd predictive SDs."""
    n = len(accepted); res = {}
    # Sample-dependent statistics (region median, marginal containment, width, ridge) use only the folds whose
    # sample is a uniform draw (quasi-random screen or a chain with split R-hat <= rhat_target).  Folds whose chain
    # did not converge keep their sample-independent results (admissibility of the true vector, non-emptiness).
    def _ok(k):
        if info is None: return True
        inf = info[k]; rh = inf.get("rhat_max", float("nan"))
        return int(inf.get("mcmc_steps", 0) or 0) == 0 or (np.isfinite(rh) and rh <= rhat_target)
    ok = [k for k in range(n) if _ok(k)]
    ne = [k for k in ok if len(accepted[k]) >= 1]
    ne2 = [k for k in ok if len(accepted[k]) >= 2]
    inv = np.array([inf["inverse_pred"] for inf in info]) if info and "inverse_pred" in info[0] else None
    for i, p in enumerate(names):
        q = {}
        if ne:
            est = np.array([np.median(accepted[k][:, i]) for k in ne])
            # quantiles taken from values present in the sample (inverted CDF), so that a small discrete region
            # (finite design) is handled correctly; for large continuous samples this is the usual central 95%
            hit = [np.quantile(accepted[k][:, i], .025, method="inverted_cdf") - 1e-7 <= X[k, i] <=
                   np.quantile(accepted[k][:, i], .975, method="inverted_cdf") + 1e-7 for k in ne]
            q.update(r2_median_accepted=_r2(X[ne, i], est), coverage95=float(np.mean(hit)), coverage95_all_folds=float(np.sum(hit)/len(ok)),
                     sd_ratio=float(np.mean([accepted[k][:, i].std() for k in ne2])/(np.sqrt(1/12) if ref_sd is None else ref_sd[i])) if ne2 else float("nan"))
        else:
            q.update(r2_median_accepted=float("nan"), coverage95=float("nan"), coverage95_all_folds=0.0, sd_ratio=float("nan"))
        if inv is not None: q["r2_inverse_regression"] = _r2(X[:, i], inv[:, i])
        res[p] = q
    n_nonempty_all = sum(1 for k in range(n) if len(accepted[k]) >= 1)
    out = {"per_param": res, "n_folds": n, "n_nonempty": n_nonempty_all, "frac_empty": float(1 - n_nonempty_all/n),
           "mean_accepted_frac": float(n_nonempty_all/n), "n_sample_ok": len(ok), "n_nonconverged": n - len(ok),
           "rhat_target": rhat_target}
    if info is not None:
        ta = np.array([inf["truth_admissible"] for inf in info])
        out["truth_admissible_all_folds"] = float(ta.mean())
        out["truth_admissible_nonempty"] = float(ta[ne].mean()) if ne else float("nan")
        vf = np.array([inf.get("volume_fraction", np.nan) for inf in info], float)
        out["volume_fraction"] = {"median": float(np.nanmedian(vf)), "min": float(np.nanmin(vf)), "max": float(np.nanmax(vf))}
        fb = [inf.get("found_by") for inf in info]
        out["found_by"] = {str(k): int(sum(f == k for f in fb)) for k in ("qmc", "grid", "design", "anneal", "walk", None)}
        rh = np.array([inf.get("rhat_max", np.nan) for inf in info], float)
        out["rhat_max_over_folds"] = float(np.nanmax(rh)) if np.isfinite(rh).any() else float("nan")
        keys = info[0]["keys"]; surf = {}
        for side in ("lo", "hi"):
            P = np.array([inf["surface_loo"][f"{side}_pred"] for inf in info]); Sd = np.array([inf["surface_loo"][f"{side}_sd"] for inf in info])
            T = np.array([inf["surface_loo"][f"{side}_true"] for inf in info])
            for q, key in enumerate(keys):
                ok = np.isfinite(P[:, q]) & np.isfinite(T[:, q])
                surf[f"{key}_{side}"] = {"loo_r2": _r2(T[ok, q], P[ok, q]) if ok.sum() > 2 else float("nan"),
                                         "within_nsd": float(np.mean(np.abs(T[ok, q] - P[ok, q]) <= n_sd*Sd[ok, q])) if ok.any() else float("nan"),
                                         "n_evaluated": int(ok.sum()),
                                         "library_unidentified": int(info[0].get("library_unidentified", [0]*len(keys))[q])}
        out["surfaces"] = surf
        out["surfaces_dropped"] = sorted({keys[q] for inf in info for q in inf.get("surfaces_dropped", [])})
        out["sampler_info"] = [{k: v for k, v in inf.items() if k not in ("surface_loo", "keys", "library_unidentified")} for inf in info]
        for k, inf in enumerate(out["sampler_info"]):          # per-fold region summaries, for later analysis without a rerun
            inf["sample_ok"] = bool(_ok(k))
            if len(accepted[k]) >= 1:
                inf["median"] = np.median(accepted[k], axis=0).tolist()
                inf["q025"] = np.quantile(accepted[k], .025, axis=0, method="inverted_cdf").tolist()
                inf["q975"] = np.quantile(accepted[k], .975, axis=0, method="inverted_cdf").tolist()
    if "init_ec" in names and "keil8" in names:
        ie, k8 = names.index("init_ec"), names.index("keil8")
        out["ridge"] = {"sd_ratio_sum": float(np.mean([(accepted[k][:, ie] + accepted[k][:, k8]).std() for k in ne2]) / np.sqrt(2/12)) if ne2 else float("nan"),
                        "sd_ratio_difference": float(np.mean([(accepted[k][:, ie] - accepted[k][:, k8]).std() for k in ne2]) / np.sqrt(2/12)) if ne2 else float("nan")}
    return out

def print_region(J, sub, label=""):
    print(f"[admissible region{(' ' + label) if label else ''}] non-empty: {J['n_nonempty']}/{J['n_folds']}; "
          f"true vector admissible: {J.get('truth_admissible_all_folds', float('nan')):.2f} of all folds", flush=True)
    if "volume_fraction" in J:
        v = J["volume_fraction"]
        print(f"  volume fraction of the box: median {v['median']:.2e} (min {v['min']:.1e}, max {v['max']:.1e}); "
              f"found by {J['found_by']}; max R-hat {J['rhat_max_over_folds']:.3f}; folds excluded from sample statistics "
              f"(chain R-hat > {J.get('rhat_target', float('nan'))}): {J.get('n_nonconverged', 0)}")
    print(f"  {'param':13s} {'R2':>7s} {'cov95':>6s} {'cov95all':>8s} {'width':>6s} {'R2 inv':>7s}")
    for p in sub:
        q = J["per_param"][p]
        print(f"  {p:13s} {q['r2_median_accepted']:7.3f} {q['coverage95']:6.2f} {q['coverage95_all_folds']:8.2f} "
              f"{q['sd_ratio']:6.2f} {q.get('r2_inverse_regression', float('nan')):7.3f}")
    if "ridge" in J: print("  ridge:", {k: round(v, 3) for k, v in J["ridge"].items()})
    if "surfaces" in J:
        s = J["surfaces"]; r2 = np.array([v["loo_r2"] for v in s.values()]); w = np.array([v["within_nsd"] for v in s.values()])
        fin = {k: v for k, v in s.items() if np.isfinite(v["loo_r2"])}
        worst = sorted(fin.items(), key=lambda kv: kv[1]["loo_r2"])[:3]
        print(f"  bound surfaces (LOO): R2 median {np.nanmedian(r2):.2f}, min {np.nanmin(r2):.2f}; held-out bound within the GP band: "
              f"median {np.nanmedian(w):.2f}, min {np.nanmin(w):.2f}; worst: " + ", ".join(f"{k} {v['loo_r2']:.2f}" for k, v in worst))
        exc = {k[:-3]: v["library_unidentified"] for k, v in s.items() if k.endswith("_lo") and v["library_unidentified"] > 0}
        if exc: print(f"  library points left out of a surface (SM parameter not identified there): {exc}")
        if J.get("surfaces_dropped"): print(f"  surfaces dropped (fewer than {MIN_LIBRARY} identified points): {J['surfaces_dropped']}")

def main():
    global SIGMA_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates-npz", required=True)   # from collect_replicates.py
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=131072)        # qmc: batch size (power of 2); walk: candidate set
    ap.add_argument("--n-cand-max", type=int, default=4194304)
    ap.add_argument("--n-min", type=int, default=300)
    ap.add_argument("--params", nargs="+", default=None)
    ap.add_argument("--n-sd", type=float, default=2.0)   # widening of interpolated bounds by GP uncertainty
    ap.add_argument("--sigma-mode", choices=["paper", "quadrature", "se"], default="paper")
    ap.add_argument("--sampler", choices=["qmc", "walk"], default="qmc")
    ap.add_argument("--folds", type=int, default=None)   # run only the first k folds (quick checks)
    ap.add_argument("--thin", type=int, default=1)       # fit the SMs on every k-th frame (temporal-correlation check)
    ap.add_argument("--rhat-target", type=float, default=1.1)
    ap.add_argument("--mcmc-max-steps", type=int, default=16000)   # cap on hit-and-run steps per walker
    ap.add_argument("--n-walkers", type=int, default=20)
    a = ap.parse_args()
    SIGMA_MODE = a.sigma_mode
    D = np.load(a.replicates_npz, allow_pickle=True)
    Y, runs, cyts = D["Y"], [str(r) for r in D["runs"]], [str(c) for c in D["cyts"]]   # Y: (n, R, T, C)
    m = json.load(open(a.manifest)); names, bounds = m["param_names"], m["bounds"]
    pmap = {r["run_id"]: r["params"] for r in m["runs"]}
    theta_all = np.array([[pmap[r][p] for p in names] for r in runs], float)
    theta = theta_all
    t = np.linspace(0, 1, Y.shape[2])
    if a.thin > 1:                                      # keep every k-th frame, time scaled on the full window
        keep = np.arange(0, Y.shape[2], a.thin); Y = Y[:, :, keep, :]; t = t[keep]
        print(f"thinning: every {a.thin}-th frame, {len(keep)} frames kept", flush=True)
    nrep = np.sum(np.isfinite(Y[:, :, 0, 0]), axis=1)
    mu = np.nanmean(Y, axis=1)
    se = np.nanstd(Y, axis=1, ddof=1) / np.sqrt(nrep)[:, None, None]
    print(f"{len(runs)} points, replicates per point: min {nrep.min()} max {nrep.max()}; sigma mode {a.sigma_mode}, sampler {a.sampler}", flush=True)
    cache = a.out + ".fits.json"
    if os.path.exists(cache):                       # stage 2 is the slow part: reuse it
        fits = [json.load(open(cache))[r] for r in runs]
        print(f"loaded SM fits and profile-likelihood bounds from {cache}", flush=True)
    else:
        jobs = [(mu[i], se[i], t, cyts, a.sigma_mode) for i in range(len(runs))]
        if a.procs > 1:
            with Pool(a.procs) as pool: fits = pool.map(fit_point, jobs)
        else:
            fits = [fit_point(j) for j in jobs]
        json.dump({r: f for r, f in zip(runs, fits)}, open(cache, "w"), default=float)
        print(f"saved SM fits to {cache}", flush=True)
    for name in CYT_SM:
        ff = [f[name] for f in fits if f.get(name)]
        r2 = [f["r2"] for f in ff]; chi2 = [f["chi2"] for f in ff]
        det = sum(f.get("deterministic", False) for f in ff)
        print(f"  SM fit {name:5s} ({CYT_SM[name]}): median R2 {np.median(r2):.3f}, min {np.min(r2):.3f}; "
              f"median chi2_min {np.median(chi2):.0f} for {Y.shape[2]} frames; deterministic branch in {det}/{len(ff)} points", flush=True)
    res = evaluate(theta, fits, mu, se, cyts, names, bounds)
    sub = a.params or names                     # ABM parameters of interest (Jain: a handful at a time)
    idx = [names.index(p) for p in sub]
    folds = list(range(a.folds)) if a.folds else None
    X, acc, info = jain_region(theta_all[:, idx], fits, runs, sub, bounds, n_cand=a.n_cand, procs=a.procs, n_sd=a.n_sd,
                               sampler=a.sampler, n_min=a.n_min, n_cand_max=a.n_cand_max, folds=folds, rhat_target=a.rhat_target,
                               mcmc_max_steps=a.mcmc_max_steps, n_walkers=a.n_walkers)
    res["jain"] = summarise_region(X, acc, sub, info, n_sd=a.n_sd, rhat_target=a.rhat_target); res["jain"]["params"] = sub
    res["jain"]["settings"] = {"sigma_mode": a.sigma_mode, "sampler": a.sampler, "n_sd": a.n_sd, "n_cand": a.n_cand,
                               "n_cand_max": a.n_cand_max, "n_min": a.n_min, "thin": a.thin, "n_frames": int(Y.shape[2]),
                               "rhat_target": a.rhat_target, "mcmc_max_steps": a.mcmc_max_steps, "n_walkers": a.n_walkers}
    print_region(res["jain"], sub)
    res["fits"] = {r: f for r, f in zip(runs, fits)}
    res["replicates_per_point"] = nrep.tolist()
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[Bergman 2024 pointwise acceptance, for reference] accepted sweep points per fold: mean {res['mean_accepted']:.1f} of {len(runs)-1}, empty in {100*res['frac_empty']:.0f}% of folds")

if __name__ == "__main__":
    main()
