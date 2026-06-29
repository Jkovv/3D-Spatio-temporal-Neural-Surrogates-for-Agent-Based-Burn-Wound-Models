import json
import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]


# A: phenomenological per-cytokine surrogate 
def _saturating(t, A, k, t0):
    """Logistic saturating curve: A / (1 + exp(-k (t - t0)))."""
    return A / (1.0 + np.exp(-k * (t - t0)))


def fit_surrogate_one(t, y):
    """Fits (A, k, t0) to one cytokine trajectory. Returns 3 surrogate params."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    tn = (t - t[0]) / max(1e-9, (t[-1] - t[0])) # normalise time to [0,1]
    A0 = max(y.max(), 1e-30)
    p0 = [A0, 8.0, 0.5]
    bounds = ([0.0, 0.0, -1.0], [10 * A0 + 1e-30, 200.0, 2.0])
    try:
        popt, _ = curve_fit(_saturating, tn, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        # fallback: crude moment estimates so the pipeline never dies on one run
        popt = [A0, 8.0, 0.5]
    return np.array(popt)


def fit_surrogates(Y, t_grid):
    n, T, C = Y.shape
    names = [f"{CYTOKINES[c]}_{p}" for c in range(C) for p in ("A", "k", "t0")]
    out = np.zeros((n, C * 3))
    for i in range(n):
        for c in range(C):
            out[i, c*3:(c+1)*3] = fit_surrogate_one(t_grid, Y[i, :, c])
    return out, names


# B: GP mapping theta_ABM -> theta_SM 
class ABMtoSMMapping:
    def __init__(self):
        self.x_scaler = None; self.y_scaler = None; self.gps = []

    def fit(self, theta_abm, theta_sm):
        self.x_scaler = StandardScaler().fit(theta_abm)
        self.y_scaler = StandardScaler().fit(theta_sm)
        Xt = self.x_scaler.transform(theta_abm)
        Yt = self.y_scaler.transform(theta_sm)
        self.gps = []
        for j in range(Yt.shape[1]):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=np.ones(Xt.shape[1]),
                               length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(1e-3, (1e-6, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8,
                                          alpha=1e-10)
            gp.fit(Xt, Yt[:, j])
            self.gps.append(gp)
        return self

    def predict_sm(self, theta_abm):
        Xt = self.x_scaler.transform(np.atleast_2d(theta_abm))
        pred = np.column_stack([gp.predict(Xt) for gp in self.gps])
        return self.y_scaler.inverse_transform(pred)


# C: inversion (recover theta_ABM from a target theta_SM) 
def recover_theta_abm(mapping, theta_sm_target, bounds_lo, bounds_hi,
                      n_restarts=12, seed=0):
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    sm_scaler = mapping.y_scaler
    tgt_scaled = sm_scaler.transform(np.atleast_2d(theta_sm_target)).ravel()

    def resid(theta_abm):
        pred = mapping.predict_sm(theta_abm)
        pred_scaled = sm_scaler.transform(pred).ravel()
        return pred_scaled - tgt_scaled

    best = None; best_cost = np.inf
    for r in range(n_restarts):
        x0 = bounds_lo + rng.random(len(bounds_lo)) * (bounds_hi - bounds_lo)
        try:
            sol = least_squares(resid, x0, bounds=(bounds_lo, bounds_hi),
                                max_nfev=2000)
        except Exception:
            continue
        if sol.cost < best_cost:
            best_cost = sol.cost; best = sol.x
    if best is None:
        best = 0.5 * (bounds_lo + bounds_hi)
    return best


def leave_one_out_recovery(theta_abm, theta_sm, param_names, bounds,
                           selected_idx=None, seed=42):
    n, p = theta_abm.shape
    idx = selected_idx if selected_idx is not None else list(range(p))
    lo = np.array([bounds[param_names[k]]["low"] for k in idx])
    hi = np.array([bounds[param_names[k]]["high"] for k in idx])

    recovered = np.zeros((n, len(idx)))
    truth = theta_abm[:, idx]
    for i in range(n):
        train = [j for j in range(n) if j != i]
        m = ABMtoSMMapping().fit(theta_abm[train][:, idx], theta_sm[train])
        recovered[i] = recover_theta_abm(m, theta_sm[i], lo, hi, seed=seed + i)

    # metrics
    rng = (hi - lo); rng[rng == 0] = 1.0
    nrmse = np.sqrt(np.mean(((recovered - truth) / rng) ** 2, axis=0))
    from sklearn.metrics import r2_score
    r2 = []
    for c in range(len(idx)):
        if np.std(truth[:, c]) > 1e-15:
            r2.append(float(r2_score(truth[:, c], recovered[:, c])))
        else:
            r2.append(float("nan"))

    return {
        "selected_params": [param_names[k] for k in idx],
        "recovered": recovered.tolist(),
        "truth": truth.tolist(),
        "nrmse_per_param": {param_names[idx[c]]: float(nrmse[c])
                            for c in range(len(idx))},
        "r2_per_param": {param_names[idx[c]]: r2[c] for c in range(len(idx))},
        "nrmse_mean": float(np.mean(nrmse)),
    }

if __name__ == "__main__":
    import argparse
    from observables import load_sweep
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selected", nargs="+", default=None,
                    help="Param names to recover (default: all).")
    ap.add_argument("--out", default="smore_recovery.json")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    names = man["param_names"]; bounds = man["bounds"]
    theta, Y, ids, t = load_sweep(args.sim_root, names)
    theta_sm, sm_names = fit_surrogates(Y, t)

    sel_idx = ([names.index(s) for s in args.selected]
               if args.selected else None)
    res = leave_one_out_recovery(theta, theta_sm, names, bounds, sel_idx)
    res["run_ids"] = ids
    json.dump(res, open(args.out, "w"), indent=2)
    print("SMoRe ParS leave-one-out recovery:")
    for pn in res["selected_params"]:
        print(f"{pn:10s}  nRMSE={res['nrmse_per_param'][pn]:.3f}  "
              f"R2={res['r2_per_param'][pn]:.3f}")
    print(f"mean nRMSE = {res['nrmse_mean']:.3f}  -> {args.out}")