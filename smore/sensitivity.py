import json
import numpy as np

from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

def _fit_gp(theta, y):
    xs = StandardScaler().fit(theta)
    ys = StandardScaler().fit(y.reshape(-1, 1))
    Xt = xs.transform(theta); yt = ys.transform(y.reshape(-1, 1)).ravel()
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(theta.shape[1]),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=10, alpha=1e-10)
    gp.fit(Xt, yt)

    def predict(theta_new):
        return ys.inverse_transform(
            gp.predict(xs.transform(theta_new)).reshape(-1, 1)).ravel()
    return predict, gp


def _salib_problem(param_names, bounds):
    return {
        "num_vars": len(param_names),
        "names": list(param_names),
        "bounds": [[bounds[n]["low"], bounds[n]["high"]] for n in param_names],
    }


def emulator_sobol(theta, Y_features, feature_names, param_names, bounds,
                   n_saltelli=1024, seed=42):
    problem = _salib_problem(param_names, bounds)
    sample = saltelli.sample(problem, n_saltelli, calc_second_order=False)

    per_feature = {}
    ST_stack = []
    for j, fname in enumerate(feature_names):
        y = Y_features[:, j]
        if np.std(y) < 1e-15:
            # constant observable -> no sensitivity signal
            continue
        predict, gp = _fit_gp(theta, y)
        y_emu = predict(sample)
        Si = sobol.analyze(problem, y_emu, calc_second_order=False,
                           print_to_console=False, seed=seed)
        per_feature[fname] = {
            "S1": Si["S1"].tolist(), "ST": Si["ST"].tolist(),
            "S1_conf": Si["S1_conf"].tolist(), "ST_conf": Si["ST_conf"].tolist(),
            "gp_loglik": float(gp.log_marginal_likelihood_value_),
        }
        ST_stack.append(np.clip(Si["ST"], 0, None))

    if not ST_stack:
        raise RuntimeError("All observables constant; no sensitivity to compute.")

    ST_mean = np.mean(ST_stack, axis=0)
    order = np.argsort(ST_mean)[::-1]
    ranking = [{"param": param_names[i], "ST_mean": float(ST_mean[i])}
               for i in order]
    return {"per_feature": per_feature, "ranking": ranking,
            "param_names": list(param_names), "n_saltelli": n_saltelli}


def select_top_k(ranking, k):
    return [r["param"] for r in ranking[:k]]


if __name__ == "__main__":
    import argparse
    from observables import load_sweep, summarize_observable, FEATURE_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="sobol_ranking.json")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    names = man["param_names"]; bounds = man["bounds"]
    theta, Y, ids, t = load_sweep(args.sim_root, names)
    feats = summarize_observable(Y)
    res = emulator_sobol(theta, feats, FEATURE_NAMES, names, bounds,
                         n_saltelli=args.n_saltelli)
    res["top_k"] = select_top_k(res["ranking"], args.top_k)
    json.dump(res, open(args.out, "w"), indent=2)
    print("Sobol ranking (mean ST across observables):")
    for r in res["ranking"]:
        print(f"  {r['param']:10s}  ST={r['ST_mean']:.4f}")
    print(f"top-{args.top_k}: {res['top_k']}")
    print(f"-> {args.out}")