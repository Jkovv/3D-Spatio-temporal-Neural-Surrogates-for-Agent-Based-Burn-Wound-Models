"""Sampling design and the init_ec x keil8 product: correlations between the sampled
parameters, the range of the product over the sweep, and a log-linear regression of
the replicate-mean IL-8 observables on log(keil8) and log(init_ec).

usage:  python smore_strict/design_check.py --manifest manifest.json --replicates-npz results/replicates.npz --out results/design_check.json
"""
import json, argparse, itertools
import numpy as np
from scipy import stats
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--replicates-npz", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = json.load(open(a.manifest)); names = m["param_names"]; base = m["baselines"]
runs = [r["run_id"] for r in m["runs"]]
th = np.array([[r["params"][p] for p in names] for r in m["runs"]], float)
ie, k8 = names.index("init_ec"), names.index("keil8")
out = {"n_points": len(runs)}

# correlations in the design
pairs = {}
for i, j in itertools.combinations(range(len(names)), 2):
    pairs[":".join(sorted((names[i], names[j])))] = {"pearson": float(stats.pearsonr(th[:, i], th[:, j])[0]),
                                       "spearman": float(stats.spearmanr(th[:, i], th[:, j])[0])}
worst = max(pairs, key=lambda k: abs(pairs[k]["pearson"]))
out["design_correlations"] = {"init_ec:keil8": pairs["init_ec:keil8"],
                              "max_abs_pearson": {"pair": worst, "value": pairs[worst]["pearson"]},
                              "all_pairs": pairs}
print(f"design: r(init_ec, keil8) = {pairs['init_ec:keil8']['pearson']:+.3f} (Spearman {pairs['init_ec:keil8']['spearman']:+.3f}); "
      f"largest |r| over {len(pairs)} pairs: {worst} {pairs[worst]['pearson']:+.3f}")

# range of the product relative to the baseline product
prod = th[:, ie]*th[:, k8]; prod_rel = prod/(base["init_ec"]*base["keil8"])
lo, hi = m["bounds"]["init_ec"], m["bounds"]["keil8"]
box_rel = (lo["low"]*hi["low"]/(base["init_ec"]*base["keil8"]), lo["high"]*hi["high"]/(base["init_ec"]*base["keil8"]))
hist, edges = np.histogram(prod_rel, bins=10, range=box_rel)
out["product"] = {"box_relative_range": box_rel, "sampled_relative_range": [float(prod_rel.min()), float(prod_rel.max())],
                  "ratio_max_min_sampled": float(prod_rel.max()/prod_rel.min()),
                  "decile_counts_over_box": hist.tolist(), "decile_edges": edges.tolist(),
                  "corr_product_init_ec": float(stats.pearsonr(prod, th[:, ie])[0]),
                  "corr_product_keil8": float(stats.pearsonr(prod, th[:, k8])[0])}
print(f"product init_ec*keil8: box {box_rel[0]:.2f}-{box_rel[1]:.2f} x baseline, sampled {prod_rel.min():.2f}-{prod_rel.max():.2f} "
      f"(ratio {prod_rel.max()/prod_rel.min():.2f}); points per decile of the box: {hist.tolist()}")

# log-linear regression of the IL-8 observables
D = np.load(a.replicates_npz, allow_pickle=True)
cyts = [str(c) for c in D["cyts"]]; rr = [str(r) for r in D["runs"]]
Y = np.nanmean(D["Y"][:, :, :, cyts.index("il8")], axis=1)                 # (n, T) replicate mean
Y = Y[[rr.index(r) for r in runs]]
obs = {"final": Y[:, -1], "mean": Y.mean(1), "max": Y.max(1), "auc": np.trapezoid(Y, axis=1)}
X2 = sm.add_constant(np.column_stack([np.log(th[:, k8]), np.log(th[:, ie])]))
Xall = sm.add_constant(np.log(th))
out["loglinear"] = {}
for name, y in obs.items():
    ly = np.log(y)
    f2 = sm.OLS(ly, X2).fit()
    ci = f2.conf_int(0.05)
    t_eq = f2.t_test("x1 - x2 = 0")                                      # beta_keil8 == beta_init_ec
    t_k8, t_ie = f2.t_test("x1 = 1"), f2.t_test("x2 = 1")
    fall = sm.OLS(ly, Xall).fit()
    fprod = sm.OLS(ly, sm.add_constant(np.log(prod))).fit()
    fadd = sm.OLS(y, sm.add_constant(th[:, [k8, ie]])).fit()
    r = {"beta_keil8": float(f2.params[1]), "ci_keil8": ci[1].tolist(), "beta_init_ec": float(f2.params[2]), "ci_init_ec": ci[2].tolist(),
         "r2": float(f2.rsquared), "p_beta_equal": float(t_eq.pvalue), "p_beta_keil8_is_1": float(t_k8.pvalue), "p_beta_init_ec_is_1": float(t_ie.pvalue),
         "r2_all_ten_log": float(fall.rsquared), "r2_log_product": float(fprod.rsquared), "r2_additive_linear": float(fadd.rsquared),
         "partial_corr_keil8_given_init_ec": float(np.sqrt(max(f2.tvalues[1]**2/(f2.tvalues[1]**2 + f2.df_resid), 0))),
         "partial_corr_init_ec_given_keil8": float(np.sqrt(max(f2.tvalues[2]**2/(f2.tvalues[2]**2 + f2.df_resid), 0)))}
    out["loglinear"][name] = r
    print(f"il8_{name:5s}: beta_keil8 {r['beta_keil8']:.5f} [{ci[1][0]:.5f}, {ci[1][1]:.5f}], beta_init_ec {r['beta_init_ec']:.5f} "
          f"[{ci[2][0]:.5f}, {ci[2][1]:.5f}], R2 {r['r2']:.8f}; p(beta equal) {r['p_beta_equal']:.3g}, p(beta_keil8=1) {r['p_beta_keil8_is_1']:.3g}, "
          f"p(beta_init_ec=1) {r['p_beta_init_ec_is_1']:.3g}; R2 all ten {r['r2_all_ten_log']:.8f}, log product {r['r2_log_product']:.8f}, "
          f"additive {r['r2_additive_linear']:.4f}")

json.dump(out, open(a.out, "w"), indent=1)
print(f"saved {a.out}")
