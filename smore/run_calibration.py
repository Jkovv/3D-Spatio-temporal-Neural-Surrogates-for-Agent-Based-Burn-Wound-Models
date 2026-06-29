import argparse
import json
from pathlib import Path

from observables import load_sweep, summarize_observable, FEATURE_NAMES
from sensitivity import emulator_sobol, select_top_k
from smore_pars import fit_surrogates, leave_one_out_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-saltelli", type=int, default=1024)
    ap.add_argument("--out", default="calibration_results.json")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    names = man["param_names"]; bounds = man["bounds"]

    print(f"[1/3] loading sweep from {args.sim_root} ...")
    theta, Y, ids, t = load_sweep(args.sim_root, names)
    feats = summarize_observable(Y)
    print(f"{len(ids)} runs | theta {theta.shape} | observable {feats.shape}")

    print(f"[2/3] emulator-based Sobol (n_saltelli={args.n_saltelli}) ...")
    sob = emulator_sobol(theta, feats, FEATURE_NAMES, names, bounds,
                         n_saltelli=args.n_saltelli)
    top_k = select_top_k(sob["ranking"], args.top_k)
    print("ranking (mean ST):")
    for r in sob["ranking"]:
        print(f"{r['param']:10s} ST={r['ST_mean']:.4f}")
    print(f"top-{args.top_k}: {top_k}")

    print(f"[3/3] SMoRe ParS leave-one-out recovery on top-{args.top_k} ...")
    theta_sm, sm_names = fit_surrogates(Y, t)
    sel_idx = [names.index(s) for s in top_k]
    rec = leave_one_out_recovery(theta, theta_sm, names, bounds, sel_idx)
    for pn in rec["selected_params"]:
        print(f"{pn:10s} nRMSE={rec['nrmse_per_param'][pn]:.3f} "
              f"R2={rec['r2_per_param'][pn]:.3f}")
    print(f"mean nRMSE = {rec['nrmse_mean']:.3f}")

    bundle = {
        "sim_root": str(args.sim_root),
        "manifest": str(args.manifest),
        "n_runs": len(ids),
        "run_ids": ids,
        "sobol": sob,
        "top_k": top_k,
        "recovery": rec,
    }
    json.dump(bundle, open(args.out, "w"), indent=2)
    print(f"\nDONE -> {args.out}")

if __name__ == "__main__":
    main()